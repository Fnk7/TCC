#include "TCC/Dialect/TC/IR/TCOps.h"
#include "TCC/Dialect/TC/Utils/Utils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::tc;

namespace {
template <typename T> inline AxesMapsTy getAxesMapsArray(T op) {
  auto axesMapsAttr = op.getAxesMaps();
  AxesMapsTy axesMaps;
  for (auto axesMapAttr : axesMapsAttr) {
    axesMaps.emplace_back(getI64ArrayFromAttr(axesMapAttr));
  }
  return axesMaps;
}
} // namespace

void MulOp::build(OpBuilder &builder, OperationState &state, ValueRange inputs,
                  AxesMapsRefTy axesMaps) {
  assert(inputs.size() >= 2 && inputs.size() == axesMaps.size());
  auto resultType = getResultTypeByAxesMaps(builder, inputs, axesMaps);
  auto axesMapsAttr = AxesMapsAttr::get(builder.getContext(), axesMaps);
  MulOp::build(builder, state, resultType, inputs, axesMapsAttr);
}
AxesMapsTy MulOp::getAxesMapsArray() { return AxesMapsTy(getAxesMaps()); }

void AddOp::build(OpBuilder &builder, OperationState &state, ValueRange inputs,
                  AxesMapsTy &axesMaps, SignsAttr signsAttr) {
  assert(inputs.size() >= 2);
  assert(inputs.size() == axesMaps.size());
  assert(inputs.size() == signsAttr.getSigns().size());
  auto resultType = getResultTypeByAxesMaps(builder, inputs, axesMaps);
  auto axesMapsAttr = AxesMapsAttr::get(builder.getContext(), axesMaps);
  AddOp::build(builder, state, resultType, inputs, axesMapsAttr, signsAttr);
}
void AddOp::build(OpBuilder &builder, OperationState &state, ValueRange inputs,
                  AxesMapsTy &axesMaps, SignsTy &signs) {
  auto signsAttr = SignsAttr::get(builder.getContext(), signs);
  AddOp::build(builder, state, inputs, axesMaps, signsAttr);
}
void AddOp::build(OpBuilder &builder, OperationState &state, ValueRange inputs,
                  AxesMapsTy &axesMaps) {
  SignsTy signs(inputs.size(), true);
  AddOp::build(builder, state, inputs, axesMaps, signs);
}
AxesMapsTy AddOp::getAxesMapsArray() { return AxesMapsTy(getAxesMaps()); }
SignsTy AddOp::getSignsArray() { return SignsTy(getSigns()); }

void ContractOp::build(OpBuilder &builder, OperationState &state, Value input,
                       AxesTy &axes) {
  std::sort(axes.begin(), axes.end());
  auto resultType = getResultTypeWithoutAxes(builder, input, axes);
  auto axesAttr = AxesAttr::get(builder.getContext(), axes);
  ContractOp::build(builder, state, resultType, input, axesAttr);
}
AxesTy ContractOp::getAxesArray() {
  auto axesRef = getAxes();
  return llvm::to_vector(axesRef);
}

void PermuteOp::build(OpBuilder &builder, OperationState &state, Value input,
                      AxesMapTy &axesMap) {
  auto resultType = getResultTypeByAxesMap(builder, input, axesMap);
  AxesMapsRefTy axesMapsRef(axesMap);
  auto axesMapAttr = AxesMapsAttr::get(builder.getContext(), axesMapsRef);
  PermuteOp::build(builder, state, resultType, input, axesMapAttr);
}
AxesMapTy PermuteOp::getAxesMapArray() { return getAxesMap()[0]; }

void ConstantOp::build(OpBuilder &builder, OperationState &state,
                       double value) {
  auto resultType = getZeroRankedTensorType(builder);
  auto valueAttr = builder.getF64FloatAttr(value);
  ConstantOp::build(builder, state, resultType, valueAttr);
}

void NegOp::build(OpBuilder &builder, OperationState &state, Value input) {
  NegOp::build(builder, state, input.getType(), input);
}

void SliceOp::build(OpBuilder &builder, OperationState &state, Value input,
                    ValueRange indices, AxesTy &axes) {
  auto resultType = getResultTypeWithoutAxes(builder, input, axes);
  auto axesAttr = AxesAttr::get(builder.getContext(), axes);
  SliceOp::build(builder, state, resultType, input, indices, axesAttr);
}

void DiagonalOp::build(OpBuilder &builder, OperationState &state, Value input,
                       AxesPairsTy &axesPairs) {
  int64_t maxAxis = -1;
  for (auto &axesPair : axesPairs) {
    for (auto axis : axesPair) {
      maxAxis = maxAxis < axis ? axis : maxAxis;
    }
  }
  auto dim = maxAxis + 1;
  AxesMapTy axesMap(dim, -1);
  for (auto e : llvm::enumerate(axesPairs)) {
    for (auto axis : e.value()) {
      assert(axesMap[axis] == -1);
      axesMap[axis] = e.index();
    }
  }
  auto resultType = getResultTypeByAxesMap(builder, input, axesMap);
  auto axesMapAttr = builder.getI64ArrayAttr(axesMap);
  SmallVector<Attribute> axesPairAttrs;
  for (auto &axesPair : axesPairs) {
    axesPairAttrs.emplace_back(builder.getI64ArrayAttr(axesPair));
  }
  auto axesPairsAttr = builder.getArrayAttr(axesPairAttrs);
  DiagonalOp::build(builder, state, resultType, input, axesMapAttr,
                    axesPairsAttr);
}

// Include TableGen files
#define GET_OP_CLASSES
#include "TCC/Dialect/TC/IR/TCOps.cpp.inc"
