#ifndef TCC_DIALECT_TC_UTILS_UTILS_H
#define TCC_DIALECT_TC_UTILS_UTILS_H

#include "TCC/Dialect/TC/IR/TCOps.h"

#include "mlir/IR/Builders.h"

template <> struct llvm::DenseMapInfo<mlir::tc::I64ArrayTy> {
  static inline mlir::tc::I64ArrayTy getEmptyKey() {
    return mlir::tc::I64ArrayTy(1, -2);
  }
  static inline mlir::tc::I64ArrayTy getTombstoneKey() {
    return mlir::tc::I64ArrayTy(1, -2);
  }
  static unsigned getHashValue(const mlir::tc::I64ArrayTy &Val) {
    return llvm::hash_combine_range(Val.begin(), Val.end());
  }
  static bool isEqual(const mlir::tc::I64ArrayTy &LHS,
                      const mlir::tc::I64ArrayTy &RHS) {
    return LHS == RHS;
  }
};

namespace mlir::tc {
// Attribute
inline I64ArrayTy getI64ArrayFromAttr(Attribute attr) {
  auto arrayAttr = attr.cast<ArrayAttr>();
  I64ArrayTy array;
  for (auto &I64 : arrayAttr) {
    array.emplace_back(I64.cast<IntegerAttr>().getValue().getSExtValue());
  }
  return array;
}
inline ArrayAttr getAxesMapsAttr(OpBuilder &builder, AxesMapsTy &axesMaps) {
  SmallVector<Attribute> axesMapAttrs;
  for (auto &axesMap : axesMaps) {
    axesMapAttrs.emplace_back(builder.getI64ArrayAttr(axesMap));
  }
  return builder.getArrayAttr(axesMapAttrs);
}
template <typename T> inline AxesMapsTy getAxesMaps(T op) {
  auto axesMapsAttr = op.getAxesMaps();
  AxesMapsTy axesMaps;
  for (auto axesMapAttr : axesMapsAttr) {
    axesMaps.emplace_back(getI64ArrayFromAttr(axesMapAttr));
  }
  return axesMaps;
}
template <> inline AxesMapsTy getAxesMaps(MulOp op) {
  return op.getAxesMapsArray();
}
template <> inline AxesMapsTy getAxesMaps(AddOp op) {
  return op.getAxesMapsArray();
}
template <typename T> inline AxesMapTy getAxesMap(T op) {
  auto axesMapAttr = op.getAxesMap();
  return getI64ArrayFromAttr(axesMapAttr);
}

// Encoded tensor type utils
int64_t getTensorRank(Value V);
int64_t getTensorRank(RankedTensorType RTT);
SizeIDsTy getTensorSizeIDs(Value V);
SizeIDsTy getTensorSizeIDs(RankedTensorType RTT);
Type getResultTypeByAxesMaps(OpBuilder &builder, ValueRange inputs,
                             AxesMapsTy &axesMaps);
Type getResultTypeWithoutAxes(OpBuilder &builder, Value input, AxesTy &axes);
Type getResultTypeByAxesMap(OpBuilder &builder, Value input,
                            AxesMapTy &axesMap);

// compare inputs order
using InputsOrder = I64ArrayTy;
using TensorSizeIDsPair = std::pair<Value, SizeIDsTy>;
InputsOrder getNewInputsOrder(ValueRange inputs);
bool compareTensorOrder(Value t1, Value t2);
bool compareTensorOrderWithSizeIDs(TensorSizeIDsPair &p1,
                                   TensorSizeIDsPair &p2);

// canonicalize axes maps
AxesMapTy canonicalizeAxesMaps(AxesMapsTy &axesMaps, int64_t dim);
inline auto getOrderedAxesMap(int64_t rank) {
  auto r = llvm::seq<int64_t>(0, rank);
  return AxesMapTy(r.begin(), r.end());
}
inline auto isOrderedAxesMap(AxesMapTy &axesMap) {
  for (auto e : llvm::enumerate(axesMap)) {
    if (e.value() != (int64_t)(e.index())) {
      return false;
    }
  }
  return true;
}
inline auto inverseAxesMap(AxesMapTy &axesMap) {
  AxesMapTy inverseMap(axesMap.size());
  for (auto e : llvm::enumerate(axesMap)) {
    inverseMap[e.value()] = e.index();
  }
  return inverseMap;
}
inline auto inverseAxesMap(AxesMapTy &axesMap, int64_t rank) {
  AxesMapTy inverseMap(rank, -1);
  for (auto e : llvm::enumerate(axesMap)) {
    inverseMap[e.value()] = e.index();
  }
  return inverseMap;
}
inline AxesMapTy getContractOpInput2OutputAxisMap(AxesTy &axes,
                                                  int64_t inputDim) {
  AxesMapTy axesMap(inputDim, 0);
  for (auto [idx, axis] : llvm::enumerate(axes)) {
    axesMap[axis] = -(1 + idx);
  }
  int64_t outAxis = 0;
  for (auto &inAxis : axesMap) {
    if (inAxis == 0) {
      inAxis = outAxis++;
    }
  }
  return axesMap;
}
inline AxesMapTy getContractOpInput2OutputAxisMap(ContractOp ctOp) {
  auto axes = ctOp.getAxesArray();
  auto inputDim = getTensorRank(ctOp.getInput());
  return getContractOpInput2OutputAxisMap(axes, inputDim);
}

inline int64_t getTensorRank(Value V) {
  auto RTT = V.getType().cast<RankedTensorType>();
  return getTensorRank(RTT);
}
inline int64_t getTensorRank(RankedTensorType RTT) { return RTT.getRank(); }
inline SizeIDsTy getTensorSizeIDs(Value V) {
  auto RTT = V.getType().cast<RankedTensorType>();
  return getTensorSizeIDs(RTT);
}
inline SizeIDsTy getTensorSizeIDs(RankedTensorType RTT) {
  auto Attr = RTT.getEncoding().cast<ArrayAttr>();
  return getI64ArrayFromAttr(Attr);
}
inline auto getTensorSizeInfos(Value V) {
  auto RTT = V.getType().cast<RankedTensorType>();
  auto &&shape = RTT.getShape();
  auto &&sizeIDs = getTensorSizeIDs(RTT);
  assert(shape.size() == sizeIDs.size());
  return std::pair(shape, sizeIDs);
}

// Fall back to only support elementwise add
inline bool isElementWiseAddOp(AddOp op) {
  auto inputs = op.getInputs();
  auto rank = getTensorRank(inputs[0]);
  for (auto input : inputs) {
    if (rank != getTensorRank(input)) {
      return false;
    }
  }
  return true;
}
} // namespace mlir::tc

#endif