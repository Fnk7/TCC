#include "TCC/Dialect/TBLAS/IR/TBLASOps.h"
#include "TCC/Dialect/TBLAS/Transforms/Passes.h"
#include "TCC/Dialect/TC/IR/TCOps.h"
#include "TCC/Dialect/TC/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"

namespace mlir::tc {
#define GEN_PASS_DEF_DEGENERALIZE
#include "TCC/Dialect/TBLAS/Transforms/Passes.h.inc"
} // namespace mlir::tc

#define DEBUG_TYPE "tblas-degeneralize"

using namespace mlir;
using namespace mlir::tc;

namespace {
Value permuteInput(OpBuilder &builder, Value input, bool &trans,
                   AxesRefTy batch, AxesRefTy contract, AxesRefTy rest) {
  AxesMapTy inversePermuteMap;
  for (auto axis : batch)
    inversePermuteMap.emplace_back(axis);
  for (auto axis : rest)
    inversePermuteMap.emplace_back(axis);
  for (auto axis : contract)
    inversePermuteMap.emplace_back(axis);
  if (isOrderedAxesMap(inversePermuteMap))
    return input;
  if (!rest.empty()) {
    AxesMapTy inversePermuteMap;
    for (auto axis : batch)
      inversePermuteMap.emplace_back(axis);
    for (auto axis : contract)
      inversePermuteMap.emplace_back(axis);
    for (auto axis : rest)
      inversePermuteMap.emplace_back(axis);
    if (isOrderedAxesMap(inversePermuteMap)) {
      trans = !trans;
      return input;
    }
  }
  auto permuteMap = inverseAxesMap(inversePermuteMap);
  auto permuteOp =
      builder.create<PermuteOp>(builder.getUnknownLoc(), input, permuteMap);
  return permuteOp.getResult();
}

void createBatchGemmFromGeneric(GenericOp op) {
  auto lhsBatch = op.getLhsBatch();
  auto rhsBatch = op.getRhsBatch();
  assert(lhsBatch.size() == rhsBatch.size());
  auto numBatch = lhsBatch.size();
  auto lhsContract = op.getLhsContract();
  auto rhsContract = op.getRhsContract();
  assert(lhsContract.size() == rhsContract.size());
  auto numK = lhsContract.size();
  auto lhsRest = op.getLhsRest();
  auto rhsRest = op.getRhsRest();
  auto numM = lhsRest.size();
  auto numN = rhsRest.size();
  OpBuilder builder(op);
  auto lhs = op.getLhs();
  auto lhsTrans = false;
  lhs = permuteInput(builder, lhs, lhsTrans, lhsBatch, lhsContract, lhsRest);
  auto rhs = op.getRhs();
  auto rhsTrans = true;
  rhs = permuteInput(builder, rhs, rhsTrans, rhsBatch, rhsContract, rhsRest);
  auto batchGemmOp =
      builder.create<BatchGemmOp>(op.getLoc(), op.getType(), lhs, rhs, numBatch,
                                  lhsTrans, rhsTrans, numM, numN, numK);
  op.replaceAllUsesWith(batchGemmOp.getResult());
  op.erase();
}

void createBatchGemvFromGeneric(GenericOp op) {
  auto lhsBatch = op.getLhsBatch();
  auto rhsBatch = op.getRhsBatch();
  assert(lhsBatch.size() == rhsBatch.size());
  auto lhsContract = op.getLhsContract();
  auto rhsContract = op.getRhsContract();
  assert(lhsContract.size() == rhsContract.size());
  auto lhsRest = op.getLhsRest();
  auto rhsRest = op.getRhsRest();
  OpBuilder builder(op);
  auto lhs = op.getLhs();
  auto lhsTrans = false;
  lhs = permuteInput(builder, lhs, lhsTrans, lhsBatch, lhsContract, lhsRest);
  auto rhs = op.getRhs();
  auto rhsTrans = false;
  rhs = permuteInput(builder, rhs, rhsTrans, rhsBatch, rhsContract, rhsRest);
  auto matrix = rhsRest.empty() ? lhs : rhs;
  auto vector = rhsRest.empty() ? rhs : lhs;
  auto numBatch = lhsBatch.size();
  auto trans = rhsRest.empty() ? lhsTrans : rhsTrans;
  auto numM = lhsRest.size() + rhsRest.size();
  auto numN = lhsContract.size();
  if (lhsTrans || rhsTrans)
    std::swap(numM, numN);
  auto batchGemvOp = builder.create<BatchGemvOp>(
      op.getLoc(), op.getType(), matrix, vector, numBatch, trans, numM, numN);
  op.replaceAllUsesWith(batchGemvOp.getResult());
  op.erase();
}

Value permuteInput(OpBuilder &builder, Value input, AxesRefTy batch,
                   AxesRefTy contract) {
  bool trans = false;
  return permuteInput(builder, input, trans, batch, contract, {});
}

void createBatchDotFromGeneric(GenericOp op) {
  auto lhsBatch = op.getLhsBatch();
  auto rhsBatch = op.getRhsBatch();
  assert(lhsBatch.size() == rhsBatch.size());
  auto lhsContract = op.getLhsContract();
  auto rhsContract = op.getRhsContract();
  assert(lhsContract.size() == rhsContract.size());
  OpBuilder builder(op);
  auto lhs = op.getLhs();
  lhs = permuteInput(builder, lhs, lhsBatch, lhsContract);
  auto rhs = op.getRhs();
  rhs = permuteInput(builder, rhs, rhsBatch, rhsContract);
  auto numBatch = lhsBatch.size();
  auto batchDotOp =
      builder.create<BatchDotOp>(op.getLoc(), op.getType(), lhs, rhs, numBatch);
  op.replaceAllUsesWith(batchDotOp.getResult());
  op.erase();
}

void degeneralizeGenericOp(GenericOp op) {
  auto lhsRest = op.getLhsRest();
  auto rhsRest = op.getRhsRest();
  if (lhsRest.empty() && rhsRest.empty()) {
    createBatchDotFromGeneric(op);
    return;
  }
  if (lhsRest.size() && lhsRest.size()) {
    createBatchGemmFromGeneric(op);
    return;
  }
  createBatchGemvFromGeneric(op);
}

struct DegeneralizePass : public tc::impl::DegeneralizeBase<DegeneralizePass> {
  void runOnOperation() override {
    SmallVector<tc::GenericOp> worklist;
    for (auto &op : getOperation().front()) {
      if (auto genericOp = dyn_cast_or_null<tc::GenericOp>(&op)) {
        worklist.emplace_back(genericOp);
      }
    }
    for (auto genericOp : worklist)
      degeneralizeGenericOp(genericOp);
  }
};
} // namespace

std::unique_ptr<Pass> mlir::tc::createTBLASDegeneralizePass() {
  return std::make_unique<DegeneralizePass>();
}
