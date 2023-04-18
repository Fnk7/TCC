#include "TCC/Dialect/TBLAS/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"

namespace mlir::tc {
#define GEN_PASS_DEF_SPECIALGEMM
#include "TCC/Dialect/TBLAS/Transforms/Passes.h.inc"
} // namespace mlir::tc

#define DEBUG_TYPE "tblas-special-gemm"

using namespace mlir;
using namespace mlir::tc;

namespace {
struct SpecialGemmPass : public tc::impl::SpecialGemmBase<SpecialGemmPass> {
  void runOnOperation() override {
    auto funcOp = getOperation();
    auto *context = &getContext();
    {
      RewritePatternSet patterns(context);
      populateSyr2kPattern(patterns, context);
      FrozenRewritePatternSet frozenPatterns(std::move(patterns));
      if (failed(applyPatternsAndFoldGreedily(funcOp, frozenPatterns)))
        return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::tc::createTBLASSpecialGemmPass() {
  return std::make_unique<SpecialGemmPass>();
}
