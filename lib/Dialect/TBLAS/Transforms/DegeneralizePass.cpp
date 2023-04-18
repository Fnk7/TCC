#include "TCC/Dialect/TBLAS/IR/TBLASOps.h"
#include "TCC/Dialect/TBLAS/Transforms/Passes.h"
#include "TCC/Dialect/TC/IR/TCOps.h"

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

struct DegeneralizePass : public tc::impl::DegeneralizeBase<DegeneralizePass> {
  void runOnOperation() override {
    // TODO
  }
};

std::unique_ptr<Pass> mlir::tc::createTBLASDegeneralizePass() {
  return std::make_unique<DegeneralizePass>();
}
