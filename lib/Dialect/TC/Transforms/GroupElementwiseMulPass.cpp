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
#define GEN_PASS_DEF_GROUPELEMENTWISEMUL
#include "TCC/Dialect/TC/Transforms/Passes.h.inc"
} // namespace mlir::tc

#define DEBUG_TYPE "tc-group-elementwise-mul"

using namespace mlir;
using namespace mlir::tc;

struct GroupElementwiseMulPass
    : public tc::impl::GroupElementwiseMulBase<GroupElementwiseMulPass> {
  void runOnOperation() override {

    auto funcOp = getOperation();
    auto *context = &getContext();
    RewritePatternSet patterns(context);
    populateGroupElementewiseMulPatterns(patterns, context);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(funcOp, frozenPatterns)))
      return signalPassFailure();
  }
};

std::unique_ptr<Pass> mlir::tc::createGroupElementwiseMulPass() {
  return std::make_unique<GroupElementwiseMulPass>();
}
