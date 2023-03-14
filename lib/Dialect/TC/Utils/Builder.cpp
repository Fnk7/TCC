#include "TCC/Dialect/TC/Utils/Builder.h"
#include "TCC/Dialect/TC/IR/TCOps.h"
#include "TCC/Dialect/TC/Utils/AxesMap.h"

namespace mlir::tc::utils {
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
