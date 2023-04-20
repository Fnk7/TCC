#include "TCC/CodeGen.h"

#include "TCC/Dialect/TBLAS/IR/TBLASOps.h"

#include "TCC/Dialect/TC/IR/TCOps.h"

#include "TCC/Dialect/TC/Utils/Utils.h"
#include "TCC/Dialect/Utils/FuncUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;
using namespace mlir::tc;
using namespace mlir::tc::utils;

namespace {
class EmitOpenCLGroupKernel : public codegen::CodeGenBase {
private:
  ModuleOp moduleOp;

public:
  EmitOpenCLGroupKernel(ModuleOp moduleOp, llvm::raw_ostream &out)
      : CodeGenBase(out), moduleOp(moduleOp) {}

  void emit() {
    auto funcOps = moduleOp.getOps<func::FuncOp>();
    for (auto funcOp : funcOps) {
      if (getWorkloadType(funcOp) == 2)
        emitWorkloadFunc(funcOp);
    }
    // TODO:
  }

private:
  void emitWorkloadFunc(func::FuncOp funcOp) {
    // TODO
    llvm::errs() << funcOp << "\n";
  }
};
} // namespace

namespace mlir::tc::codegen {
void emitTBLASToOpenCL(ModuleOp op, llvm::raw_ostream &out, bool workgroup) {
  if (!workgroup)
    llvm_unreachable("TODO: emit tblas to OpenCL kernel.\n");
  EmitOpenCLGroupKernel(op, out).emit();
}
} // namespace mlir::tc::codegen
