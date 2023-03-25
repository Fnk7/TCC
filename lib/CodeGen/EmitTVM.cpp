#include "TCC/CodeGen.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;
using namespace mlir::tc;

namespace {
class EmitTVMImpl : public codegen::CodeGenBase {
  func::FuncOp funcOp;
  Block &block;

public:
  EmitTVMImpl(func::FuncOp funcOp, llvm::raw_ostream &out)
      : CodeGenBase(out), funcOp(funcOp), block(funcOp.front()) {}

  void emit() {
    /// TODO:
    out << "TODO\n";
  }

private:
  // TODO
};
} // namespace

namespace mlir::tc::codegen {
void emitTCToTVM(ModuleOp op, llvm::raw_ostream &out) {
  SymbolTable symTable(op);
  auto funcOps = op.getOps<func::FuncOp>();
  for (auto funcOp : funcOps) {
    EmitTVMImpl(funcOp, out).emit();
  }
}
} // namespace mlir::tc::codegen
