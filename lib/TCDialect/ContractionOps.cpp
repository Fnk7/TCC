#include "TCDialect/ContractionOps.h"

using namespace mlir;
using namespace mlir::contraction;

#include "TCDialect/ContractionOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

void ContractDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TCDialect/ContractionOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Ops
//===----------------------------------------------------------------------===//

void IndexOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    mlir::Value tensor, mlir::ValueRange indexes) {
  state.addTypes(builder.getF64Type());
  state.addOperands({tensor});
  state.addOperands(indexes);
}

void AddOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(builder.getF64Type());
  state.addOperands({lhs, rhs});
}

void MulOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(builder.getF64Type());
  state.addOperands({lhs, rhs});
}

void ForOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::ValueRange ranges) {
  state.addTypes(mlir::RankedTensorType::get(
      SmallVector<int64_t>(ranges.size(), -1), builder.getF64Type()));
  state.addOperands(ranges);
  Region *region = state.addRegion();
  region->push_back(new Block);
  Block &block = region->front();
  for (Value v : ranges)
    block.addArgument(v.getType(), v.getLoc());
}

void ContractOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       mlir::ValueRange ranges) {
  state.addTypes(builder.getF64Type());
  state.addOperands(ranges);
  Region *region = state.addRegion();
  region->push_back(new Block);
  Block &block = region->front();
  for (Value v : ranges)
    block.addArgument(v.getType(), v.getLoc());
}

void ForOp::getSuccessorRegions(Optional<unsigned> index,
                                ArrayRef<Attribute> operands,
                                SmallVectorImpl<RegionSuccessor> &regions) {
  return;
}

void ContractOp::getSuccessorRegions(
    Optional<unsigned> index, ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &regions) {
  return;
}

#define GET_OP_CLASSES
#include "TCDialect/ContractionOps.cpp.inc"
