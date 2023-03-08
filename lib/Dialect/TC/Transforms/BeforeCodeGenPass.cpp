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
#define GEN_PASS_DEF_BEFORECODEGEN
#include "TCC/Dialect/TC/Transforms/Passes.h.inc"
} // namespace mlir::tc

#define DEBUG_TYPE "tc-before-codegen"

using namespace mlir;
using namespace mlir::tc;

struct BeforeCodeGenPass
    : public tc::impl::BeforeCodeGenBase<BeforeCodeGenPass> {
  void runOnOperation() override {

    auto funcOp = getOperation();
    auto *context = &getContext();
    RewritePatternSet patterns(context);
    populateSimpilfyCodeGenPatterns(patterns, context);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(funcOp, frozenPatterns)))
      return signalPassFailure();
  }
};

std::unique_ptr<Pass> mlir::tc::createBeforeCodeGenPass() {
  return std::make_unique<BeforeCodeGenPass>();
}
