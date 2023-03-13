#include "TCC/Dialect/TC/Utils/Debug.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::tc::utils {
void dumpValueDefiningChain(Value v, llvm::SetVector<Value> &dumpped) {
  if (dumpped.count(v)) {
    return;
  }
  dumpped.insert(v);
  if (auto arg = v.dyn_cast<BlockArgument>()) {
    auto funcOp = arg.getParentRegion()->getParentOfType<func::FuncOp>();
    auto name = funcOp.getArgAttr(arg.getArgNumber(), "func.name");
    if (name) {
      llvm::errs() << "\%arg" << arg.getArgNumber() << " -> "
                   << name.cast<StringAttr>().getValue() << " ";
      arg.getType().dump();
      llvm::errs() << "\n";
    } else {
      arg.dump();
    }
  } else if (auto *op = v.getDefiningOp()) {
    for (auto v : op->getOperands()) {
      dumpValueDefiningChain(v, dumpped);
    }
    op->dump();
  }
}

void dumpValueDefiningChain(Value v) {
  llvm::SetVector<Value> dumpped;
  dumpValueDefiningChain(v, dumpped);
}
} // namespace mlir::tc::utils
