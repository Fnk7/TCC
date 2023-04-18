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
  {
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

void degeneralizeGenericOp(GenericOp op) {
  auto lhsRest = op.getLhsRest();
  auto rhsRest = op.getRhsRest();
  if (lhsRest.empty() && rhsRest.empty()) {
    llvm_unreachable("TODO: generate batch vector vector dot");
    return;
  }
  if (lhsRest.size() && lhsRest.size()) {
    createBatchGemmFromGeneric(op);
    return;
  }
  assert("TODO: generate batch matrix vector dot");
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
