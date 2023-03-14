#include "TCC/Dialect/TC/Utils/Builder.h"
#include "TCC/Dialect/TC/IR/TCOps.h"
#include "TCC/Dialect/TC/Utils/AxesMap.h"

namespace mlir::tc::utils {
int64_t getRankFromAxesMaps(AxesMapsTy &axesMaps) {
  int64_t maxAxis = -1;
  for (auto &axesMap : axesMaps) {
    for (auto axis : axesMap) {
      maxAxis = std::max(axis, maxAxis);
    }
  }
  return maxAxis + 1;
}
void mapOriginAxis2NewAxis(AxesMapTy &axesMap) {
  int64_t newAxis = 0;
  for (auto &axis : axesMap) {
    if (axis >= 0) {
      axis = newAxis++;
    }
  }
}
AxesMapTy getOrigin2NewAxesMap(AxesMapsTy &axesMaps) {
  auto rank = getRankFromAxesMaps(axesMaps);
  AxesMapTy origin2NewMap(rank, -1);
  for (auto &axesMap : axesMaps) {
    for (auto axis : axesMap) {
      origin2NewMap[axis] = axis;
    }
  }
  mapOriginAxis2NewAxis(origin2NewMap);
  return origin2NewMap;
}
AxesMapTy getNew2OriginAxesMap(AxesMapTy &origin2NewMap) {
  AxesMapTy new2OriginMap;
  for (auto [originAxis, newAxis] : llvm::enumerate(origin2NewMap)) {
    if (newAxis >= 0) {
      new2OriginMap.emplace_back(originAxis);
    }
  }
  return new2OriginMap;
}
MappedInputTy createMulOp(OpBuilder &builder, MappedInputsTy &mappedInputs) {
  SmallVector<Value> inputs;
  AxesMapsTy axesMaps;
  for (auto [input, axesMap] : mappedInputs) {
    inputs.emplace_back(input);
    axesMaps.emplace_back(axesMap);
  }
  return createMulOp(builder, inputs, axesMaps);
}
MappedInputTy createMulOp(OpBuilder &builder, ValueRange inputs,
                          AxesMapsTy &axesMaps) {
  if (inputs.size() == 1) {
    return MappedInputTy(inputs[0], axesMaps[0]);
  }
  AxesMapTy origin2NewMap = getOrigin2NewAxesMap(axesMaps);
  applyAxesMap(origin2NewMap, axesMaps);
  auto mulOp = builder.create<MulOp>(builder.getUnknownLoc(), inputs, axesMaps);
  auto retAxesMap = getNew2OriginAxesMap(origin2NewMap);
  return MappedInputTy(mulOp.getResult(), retAxesMap);
}
MappedInputTy createAddOp(OpBuilder &builder, MappedInputsTy &mappedInputs) {
  SmallVector<Value> inputs;
  AxesMapsTy axesMaps;
  for (auto [input, axesMap] : mappedInputs) {
    inputs.emplace_back(input);
    axesMaps.emplace_back(axesMap);
  }
  return createAddOp(builder, inputs, axesMaps);
}
MappedInputTy createAddOp(OpBuilder &builder, ValueRange inputs,
                          AxesMapsTy &axesMaps) {
  if (inputs.size() == 1) {
    return MappedInputTy(inputs[0], axesMaps[0]);
  }
  AxesMapTy origin2NewMap = getOrigin2NewAxesMap(axesMaps);
  applyAxesMap(origin2NewMap, axesMaps);
  auto mulOp = builder.create<AddOp>(builder.getUnknownLoc(), inputs, axesMaps);
  auto retAxesMap = getNew2OriginAxesMap(origin2NewMap);
  return MappedInputTy(mulOp.getResult(), retAxesMap);
}
} // namespace mlir::tc::utils
