#include "TCC/Dialect/TBLAS/IR/TBLASOps.h"
#include "TCC/Dialect/TBLAS/Utils/Utils.h"

#include "TCC/Dialect/TC/Utils/Utils.h"

#include "mlir/IR/Builders.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::tc;
using namespace mlir::tc;

// void GenericOp::build(OpBuilder &builder, OperationState &state, Type result,
//                       Value lhs, Value rhs, AxesRefTy lhs_batch,
//                       AxesRefTy rhs_batch, AxesRefTy lhs_rest,
//                       AxesRefTy rhs_rest, AxesRefTy lhs_contract,
//                       AxesRefTy rhs_contract) {
//   auto lhsBatchAttr = builder.getDenseI64ArrayAttr(lhs_batch);
//   auto rhsBatchAttr = builder.getDenseI64ArrayAttr(rhs_batch);
//   auto lhsRestAttr = builder.getDenseI64ArrayAttr(lhs_rest);
//   auto rhsRestAttr = builder.getDenseI64ArrayAttr(rhs_rest);
//   auto lhsContractAttr = builder.getDenseI64ArrayAttr(lhs_contract);
//   auto rhsContractAttr = builder.getDenseI64ArrayAttr(rhs_contract);
//   GenericOp::build(builder, state, result, lhs, rhs, lhsBatchAttr, rhsBatchAttr,
//                    lhsRestAttr, rhsRestAttr, lhsContractAttr, rhsContractAttr);
// }

void BatchMatMulOp::build(OpBuilder &builder, OperationState &state,
                          Value input1, Value input2, AxesTy &batch_axes1,
                          AxesTy &batch_axes2, AxesTy &rest_axes1,
                          AxesTy &rest_axes2, AxesTy &contract_axes1,
                          AxesTy &contract_axes2) {
  auto [shape1, sizeIDs1] = tc::getTensorSizeInfos(input1);
  auto [shape2, sizeIDs2] = tc::getTensorSizeInfos(input2);
  I64ArrayTy shape;
  tc::SizeIDsTy sizeIDs;
  for (auto axis : batch_axes1) {
    shape.emplace_back(shape1[axis]);
    sizeIDs.emplace_back(sizeIDs1[axis]);
  }
  for (auto axis : rest_axes1) {
    shape.emplace_back(shape1[axis]);
    sizeIDs.emplace_back(sizeIDs1[axis]);
  }
  for (auto axis : rest_axes2) {
    shape.emplace_back(shape2[axis]);
    sizeIDs.emplace_back(sizeIDs2[axis]);
  }
  auto resultType = tc::getEncodedTensorType(builder, shape, sizeIDs);
  auto batchAxes1Attr = builder.getI64ArrayAttr(batch_axes1);
  auto batchAxes2Attr = builder.getI64ArrayAttr(batch_axes2);
  auto restAxes1Attr = builder.getI64ArrayAttr(rest_axes1);
  auto restAxes2Attr = builder.getI64ArrayAttr(rest_axes2);
  auto contractAxes1Attr = builder.getI64ArrayAttr(contract_axes1);
  auto contractAxes2Attr = builder.getI64ArrayAttr(contract_axes2);
  BatchMatMulOp::build(builder, state, resultType, input1, input2,
                       batchAxes1Attr, batchAxes2Attr, restAxes1Attr,
                       restAxes2Attr, contractAxes1Attr, contractAxes2Attr);
}

// Include TableGen files
#define GET_OP_CLASSES
#include "TCC/Dialect/TBLAS/IR/TBLASOps.cpp.inc"
