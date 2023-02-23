#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

#include "TCC/Dialect/TC/IR/TCOps.h"
#include "TCC/Dialect/TC/Utils/Utils.h"
#include "TCC/Passes.h"

#define DEBUG_TYPE "return-2-argument"

using namespace mlir;
using namespace mlir::tc;

namespace {
struct ReturnToArgumentPass
    : public mlir::PassWrapper<ReturnToArgumentPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReturnToArgumentPass)

  void runOnFuncOp(func::FuncOp funcOp) {
    auto &entryBlock = funcOp.front();
    auto returnOp = cast<func::ReturnOp>(entryBlock.getTerminator());
    auto retValues = returnOp.getOperands();
    auto funcAttr = funcOp->getAttrDictionary();
    auto writebacksAttr = funcAttr.get("wirtebacks").cast<ArrayAttr>();
    auto writebacks = getI64ArrayFromAttr(writebacksAttr);
    SmallVector<Type> inputsType(funcOp.getArgumentTypes());
    for (auto [wb, retValue] : llvm::zip(writebacks, retValues)) {
      if (wb < 0) {
        auto retArgType = retValue.getType();
        inputsType.emplace_back(retArgType);
        auto retArg = entryBlock.addArgument(retArgType, retValue.getLoc());
        retValue.replaceAllUsesWith(retArg);
      } else {
        OpBuilder builder(returnOp);
        auto retArg = entryBlock.getArgument(wb);
        builder.create<memref::CopyOp>(retValue.getLoc(), retValue, retArg);
      }
    }
    funcOp.setType(FunctionType::get(&getContext(), inputsType, {}));
    returnOp->eraseOperands(0, returnOp.getNumOperands());
  }

  void runOnOperation() override {
    getOperation().walk([&](func::FuncOp op) { this->runOnFuncOp(op); });
  }
};
} // namespace

std::unique_ptr<mlir::Pass> mlir::tc::createReturnToArgumentPass() {
  return std::make_unique<ReturnToArgumentPass>();
}
