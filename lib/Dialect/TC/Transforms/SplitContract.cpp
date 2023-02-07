#include "TCC/Dialect/TC/IR/TCOps.h"
#include "TCC/Dialect/TC/Transforms/Passes.h"
#include "TCC/Dialect/TC/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "llvm/Support/Debug.h"

namespace mlir::tc {
#define GEN_PASS_DEF_SPLITCONTRACT
#include "TCC/Dialect/TC/Transforms/Passes.h.inc"
} // namespace mlir::tc

#define DEBUG_TYPE "split-contract"

using namespace mlir;
using namespace mlir::tc;

void splitContractOpWithMulOp(ContractOp op) {
  auto input = op.getInput();
  auto axes = getAxes(op);
  auto mulOp = input.getDefiningOp<MulOp>();
  if (axes.size() < 2 || !mulOp) {
    return;
  }
  OpBuilder builder(op);
  std::sort(axes.begin(), axes.end());
  for (auto axis : llvm::reverse(axes)) {
    AxesTy newAxes(1, axis);
    auto newOp = builder.create<ContractOp>(op.getLoc(), input, newAxes);
    input = newOp.getResult();
  }
  op.replaceAllUsesWith(input);
  op.erase();
}

struct SplitContractPass
    : public tc::impl::SplitContractBase<SplitContractPass> {
  void runOnOperation() override {
    auto funcOp = getOperation();
    auto &region = funcOp.getRegion();
    auto &entryBlock = *(region.getBlocks().begin());
    SmallVector<ContractOp> workLists;
    for (auto &op : entryBlock) {
      if (auto contractOp = dyn_cast_or_null<ContractOp>(&op)) {
        workLists.emplace_back(contractOp);
      }
    }
    for (auto contractOp : workLists) {
      splitContractOpWithMulOp(contractOp);
    }
  }
};

std::unique_ptr<Pass> mlir::tc::createSplitContractPass() {
  return std::make_unique<SplitContractPass>();
}
