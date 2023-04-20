#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Pass/Pass.h"

#include "TCC/Dialect/TBLAS/IR/TBLASOps.h"
#include "TCC/Dialect/TC/IR/TCOps.h"
#include "TCC/Dialect/TC/Utils/Utils.h"
#include "TCC/Dialect/Utils/FuncUtils.h"
#include "TCC/Passes.h"

#define DEBUG_TYPE "tc-split-workload"

using namespace mlir;
using namespace mlir::tc;

namespace {
struct TCDCEPass
    : public mlir::PassWrapper<TCDCEPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TCDCEPass)

  void runOnOperation() override {
    auto &block = getOperation().front();
    SmallVector<Operation *> toErase;
    do {
      toErase.clear();
      for (auto &op : block) {
        if (op.hasTrait<OpTrait::IsTerminator>() ||
            !MemoryEffectOpInterface::hasNoEffect(&op))
          continue;
        bool erase = true;
        for (auto result : op.getResults()) {
          if (!result.getUsers().empty()) {
            erase = false;
            break;
          }
        }
        if (erase)
          toErase.emplace_back(&op);
      }
      for (auto op : toErase)
        op->erase();
    } while (toErase.size());
  }
};
} // namespace

std::unique_ptr<mlir::Pass> mlir::tc::createTCDCEPass() {
  return std::make_unique<TCDCEPass>();
}
