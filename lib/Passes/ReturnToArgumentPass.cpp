#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

#include "TCC/Dialect/ContractionOps.h"
#include "TCC/Passes.h"

#define DEBUG_TYPE "return-2-argument"

using namespace mlir;

namespace {
struct ReturnToArgumentPass
    : public mlir::PassWrapper<ReturnToArgumentPass,
                               OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReturnToArgumentPass)

  void runOnOperation() override {
    auto Func = getOperation();
    auto &EntryB = Func.front();
    auto Return = cast<func::ReturnOp>(EntryB.getTerminator());
    auto ReturnValues = Return.getOperands();
    SmallVector<Type> InputTypes(Func.getArgumentTypes());
    for (auto V : ReturnValues) {
      auto Alloc = cast<memref::AllocOp>(V.getDefiningOp());
      auto ArgType = Alloc.getType();
      InputTypes.emplace_back(ArgType);
      auto Arg = EntryB.addArgument(ArgType, Alloc.getLoc());
      Alloc.replaceAllUsesWith(Arg);
    }
    Func.setType(FunctionType::get(&getContext(), InputTypes, {}));
    Return->eraseOperands(0, Return.getNumOperands());
  }
};
} // namespace

std::unique_ptr<mlir::Pass> mlir::contraction::createReturnToArgumentPass() {
  return std::make_unique<ReturnToArgumentPass>();
}
