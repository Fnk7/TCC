#ifndef TCC_DIALECT_TC_UTILS_UTILS_H
#define TCC_DIALECT_TC_UTILS_UTILS_H

#include "TCC/Dialect/TC/IR/TCOps.h"

#include "mlir/IR/Builders.h"

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
template <typename T> inline AxesMapTy getAxesMap(T op) {
  auto axesMapAttr = op.getAxesMap();
  return getI64ArrayFromAttr(axesMapAttr);
}
template <typename T> inline AxesTy getAxes(T op) {
  auto axesAttr = op.getAxes();
  return getI64ArrayFromAttr(axesAttr);
}
inline SignsTy getSigns(AddOp op) {
  SignsTy signs;
  auto signsAttr = op.getSigns();
  for (auto signAttr : signsAttr) {
    signs.emplace_back(signAttr.cast<BoolAttr>().getValue());
  }
  return signs;
}

// Encoded tensor type utils
Type getEncodedTensorType(OpBuilder &builder, TShapeTy &shape,
                          SizeIDsTy &encoding);
Type getEncodedTensorType(OpBuilder &builder, ArrayRef<int64_t> shape,
                          SizeIDsTy &encoding);
Type getZeroRankedTensorType(OpBuilder &builder);
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
inline AxesMapTy getContractOpInput2OutputAxisMap(AxesTy &axes,
                                                  int64_t inputDim) {
  AxesMapTy axesMap(inputDim, 0);
  for (auto axis : axes) {
    axesMap[axis] = -1;
  }
  int64_t outAxis = 0;
  for (auto &inAxis : axesMap) {
    if (inAxis == 0) {
      inAxis = outAxis++;
    }
  }
  return axesMap;
}

inline Type getEncodedTensorType(OpBuilder &builder, TShapeTy &shape,
                                 SizeIDsTy &encoding) {
  auto encodingAttr = builder.getI64ArrayAttr(encoding);
  return RankedTensorType::get(shape, builder.getF64Type(), encodingAttr);
}
inline Type getEncodedTensorType(OpBuilder &builder, ArrayRef<int64_t> shape,
                                 SizeIDsTy &encoding) {
  auto encodingAttr = builder.getI64ArrayAttr(encoding);
  return RankedTensorType::get(shape, builder.getF64Type(), encodingAttr);
}
inline Type getZeroRankedTensorType(OpBuilder &builder) {
  TShapeTy shape;
  SizeIDsTy sizeIDs;
  return getEncodedTensorType(builder, shape, sizeIDs);
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