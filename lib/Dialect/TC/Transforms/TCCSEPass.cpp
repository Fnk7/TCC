#include "TCC/Dialect/TC/IR/TCOps.h"
#include "TCC/Dialect/TC/IR/TCPatterns.h"
#include "TCC/Dialect/TC/Transforms/Passes.h"
#include "TCC/Dialect/TC/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"

namespace mlir::tc {
#define GEN_PASS_DEF_TCCSE
#include "TCC/Dialect/TC/Transforms/Passes.h.inc"
} // namespace mlir::tc

#define DEBUG_TYPE "tc-cse"

using namespace mlir;
using namespace mlir::tc;

struct TCCSEPass : public tc::impl::TCCSEBase<TCCSEPass> {
  void runOnOperation() override {

    auto funcOp = getOperation();
    Block block;
    auto opNum = funcOp.front().getOperations().size();
    auto *context = &getContext();
    RewritePatternSet patterns(context);
    getCanonicalizeForCSEPatterns(patterns, context);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    int iter = 0, maxIter = 4;
    while (true) {
      bool toBreak = true;
      if (failed(applyPatternsAndFoldGreedily(funcOp, frozenPatterns)))
        return signalPassFailure();
      {
        auto opNum2 = funcOp.front().getOperations().size();
        if (opNum != opNum2) {
          toBreak = false;
          opNum = opNum2;
        }
      }
      OpPassManager dynamicPM;
      dynamicPM.addPass(createCSEPass());
      if (failed(runPipeline(dynamicPM, funcOp)))
        return signalPassFailure();
      {
        auto opNum2 = funcOp.front().getOperations().size();
        if (opNum != opNum2) {
          toBreak = false;
          opNum = opNum2;
        }
      }
      if (toBreak) {
        break;
      }
      if (iter++ == maxIter) {
        return signalPassFailure();
      }
    }
  }
};

std::unique_ptr<Pass> mlir::tc::createTCCSEPass() {
  return std::make_unique<TCCSEPass>();
}
