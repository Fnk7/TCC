#include "TCC/Dialect/TC/Utils/Utils.h"
#include "TCC/Dialect/TC/IR/TCOps.h"

namespace mlir::tc {
// TODO: delete after debug
auto &operator<<(raw_ostream &OS, ArrayRef<int64_t> i64Array) {
  for (auto i : i64Array)
    OS << i << " ";
  return OS;
}
// Encoded tensor type utils
Type getResultTypeByAxesMaps(OpBuilder &builder, ValueRange inputs,
                             AxesMapsTy &axesMaps) {
  int64_t maxAxis = -1;
  for (auto &axesMap : axesMaps) {
    for (auto axis : axesMap) {
      maxAxis = maxAxis < axis ? axis : maxAxis;
    }
  }
  auto dim = maxAxis + 1;
  TShapeTy shape(dim);
  SizeIDsTy sizeIDs(dim, -1);
  for (auto [input, axesMap] : llvm::zip(inputs, axesMaps)) {
    auto [inputShape, inputSizeIDs] = getTensorSizeInfos(input);
    for (auto e : llvm::enumerate(axesMap)) {
      if (sizeIDs[e.value()] != -1) {
        assert(shape[e.value()] == inputShape[e.index()]);
        assert(sizeIDs[e.value()] == inputSizeIDs[e.index()]);
      } else {
        shape[e.value()] = inputShape[e.index()];
        sizeIDs[e.value()] = inputSizeIDs[e.index()];
      }
    }
  }
  for (auto sizeID : sizeIDs) {
    assert(sizeID != -1);
  }
  return getEncodedTensorType(builder, shape, sizeIDs);
}
Type getResultTypeWithoutAxes(OpBuilder &builder, Value input, AxesTy &axes) {
  auto [inputShape, inputSizeIDs] = getTensorSizeInfos(input);
  assert(inputSizeIDs.size() >= axes.size());
  SmallVector<bool> contracted(inputSizeIDs.size(), false);
  for (auto axis : axes) {
    contracted[axis] = true;
  }
  TShapeTy shape;
  SizeIDsTy sizeIDs;
  for (auto [c, size, sizeID] :
       llvm::zip(contracted, inputShape, inputSizeIDs)) {
    if (!c) {
      shape.emplace_back(size);
      sizeIDs.emplace_back(sizeID);
    }
  }
  return getEncodedTensorType(builder, shape, sizeIDs);
}
Type getResultTypeByAxesMap(OpBuilder &builder, Value input,
                            AxesMapTy &axesMap) {
  int64_t maxAxis = -1;
  for (auto axis : axesMap) {
    maxAxis = maxAxis < axis ? axis : maxAxis;
  }
  auto dim = maxAxis + 1;
  TShapeTy shape(dim);
  SizeIDsTy sizeIDs(dim, -1);
  auto [inputShape, inputSizeIDs] = getTensorSizeInfos(input);
  for (auto e : llvm::enumerate(axesMap)) {
    shape[e.value()] = inputShape[e.index()];
    sizeIDs[e.value()] = inputSizeIDs[e.index()];
  }
  for (auto sizeID : sizeIDs) {
    if (sizeID == -1) {
      input.getType().dump();
      llvm::errs() << "\n" << axesMap << "\n";
      llvm::errs() << sizeIDs << '\n';
      assert(sizeID != -1);
    }
  }
  return getEncodedTensorType(builder, shape, sizeIDs);
}

// compare inputs order
InputsOrder getNewInputsOrder(ValueRange inputs) {
  using SortItem = std::pair<int64_t, TensorSizeIDsPair>;
  SmallVector<SortItem> items;
  for (auto e : llvm::enumerate(inputs)) {
    auto input = e.value();
    auto &&pair = TensorSizeIDsPair(input, getTensorSizeIDs(input));
    auto &&item = SortItem(e.index(), pair);
    items.emplace_back(item);
  }
  std::sort(items.begin(), items.end(), [](SortItem &i1, SortItem &i2) {
    return compareTensorOrderWithSizeIDs(i1.second, i2.second);
  });
  InputsOrder order;
  for (auto item : items) {
    order.emplace_back(item.first);
  }
  // bug record : inputs[e.value()] == inputs[e.index()]
  // bug example : [%arg0,%arg0,%arg1], [1,0,2]
  for (auto e : llvm::enumerate(order)) {
    if (e.value() != (int64_t)e.index() &&
        inputs[e.value()] != inputs[e.index()]) {
      return order;
    }
  }
  order.clear();
  return order;
}

bool compareTensorOrder(Value t1, Value t2) {
  if (t1 == t2) {
    return false;
  }
  if (auto b1 = t1.dyn_cast<BlockArgument>()) {
    if (auto b2 = t2.dyn_cast<BlockArgument>()) {
      if (b1.getArgNumber() > b2.getArgNumber())
        return false;
    }
    return true;
  } else if (t2.isa<BlockArgument>()) {
    return false;
  }
  return t1.getDefiningOp()->isBeforeInBlock(t2.getDefiningOp());
}
bool compareTensorOrderWithSizeIDs(TensorSizeIDsPair &p1,
                                   TensorSizeIDsPair &p2) {
  auto t1 = p1.first;
  auto t2 = p2.first;
  auto &sizeIDs1 = p1.second;
  auto &sizeIDs2 = p2.second;
  if (t1 == t2) {
    return false;
  }
  auto minDim = std::min(sizeIDs1.size(), sizeIDs2.size());
  for (auto idx : llvm::seq<int64_t>(0, minDim)) {
    if (sizeIDs1[idx] < sizeIDs2[idx])
      return true;
    if (sizeIDs1[idx] > sizeIDs2[idx])
      return false;
  }
  if (sizeIDs1.size() < sizeIDs2.size()) {
    return true;
  }
  if (sizeIDs1.size() > sizeIDs2.size()) {
    return false;
  }
  return compareTensorOrder(t1, t2);
}

// canonicalize axes maps
AxesMapTy canonicalizeAxesMaps(AxesMapsTy &axesMaps, int64_t dim) {
  /* output sizeIDs [s0,s1,s2,s3]
   * input  sizeIDs [[s1,s3,s2],[s1,s0,s2]]
   * axesMaps     [[a1,a3,a2],[a1,a0,a2]]
   * new AxesMaps [[b0,b1,b2],[b0,b3,b2]]
   * new output sizeIDs [s1,s3,s2,s0]
   * bmap = [3,0,2,1]; b = bmap[a]; inverseMap
   * sizeIDa[amap[b]] = sizeIDb[b]
   * amap = [1,3,2,0]; a = amap[b]; permuteMap
   */
  // output sizeIDs -> new output sizeIDs
  AxesMapTy inverseMap(dim, -1);
  int64_t axis2 = 0;
  for (auto &axesMap : axesMaps) {
    for (auto &axis : axesMap) {
      if (inverseMap[axis] == -1) {
        inverseMap[axis] = axis2;
        axis2++;
      }
      axis = inverseMap[axis];
    }
  }
  return inverseMap;
}

/// Create RankedTensorType with shape, sizeIDs and symmetry
Type getEncodedTensorType(OpBuilder &builder, ShapeRefTy shape,
                          SizeIDsRefTy sizeIDs, SymmArrayRefTy symmetry) {
  auto symmetryAttr = SymmetryAttr::get(builder.getContext(), symmetry);
  auto encoding = SmallVector<NamedAttribute>(
      {builder.getNamedAttr("size_id", builder.getDenseI64ArrayAttr(sizeIDs)),
       builder.getNamedAttr("symmetry", symmetryAttr)});
  auto encodingAttr = builder.getDictionaryAttr(encoding);
  return RankedTensorType::get(shape, builder.getF64Type(), encodingAttr);
}
} // namespace mlir::tc
