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
    for (auto funcOp : funcOps) {
      if (getWorkloadType(funcOp) == 1)
        emitEntryFunc(funcOp);
    }
  }

private:
  int indent = 0;
  using ValueNameMapTy = DenseMap<Value, std::string>;
  ValueNameMapTy valueNameMap;
  using SizeIDValueMapTy = DenseMap<int64_t, std::string>;
  SizeIDValueMapTy sizeIDValueMap;

  void emitFunctionType(func::FuncOp funcOp) {
    std::string funcDeclare = "__kernel void ";
    out << funcDeclare;
    indent = funcDeclare.size();
    out << funcOp.getName();
    indent += funcOp.getName().size();
    out << "(";
    indent += 1;
    bool newLine = false;
    emitArgumentTypes(funcOp.front(), newLine);
    emitResultTypes(funcOp.front(), newLine);
    out << ")\n";
  }

  void emitArgumentTypes(Block &block, bool &newLine) {
    auto inputs = block.getArguments();
    for (auto [idx, input] : llvm::enumerate(inputs)) {
      if (newLine) {
        out << ",\n";
        out.indent(indent);
      }
      newLine = true;
      auto name = "arg" + std::to_string(idx);
      if (input.getType().isIndex()) {
        out << "int " << name;
        sizeIDValueMap[idx] = name;
      } else {
        out << "double* " << name;
      }
      valueNameMap[input] = name;
    }
  }

  SmallVector<std::string> returnNames;
  void emitResultTypes(Block &block, bool &newLine) {
    auto returnOp = block.getTerminator();
    for (auto [idx, input] : llvm::enumerate(returnOp->getOperands())) {
      if (newLine) {
        out << ",\n";
        out.indent(indent);
      }
      newLine = true;
      auto name = "ret" + std::to_string(idx);
      out << "double* " << name;
      valueNameMap[input] = name;
      returnNames.emplace_back(name);
    }
  }

  void emitWorkloadFunctionBody(func::FuncOp funcOp) {
    out << "{\n";
    indent = 2;
    out << "}\n";
  }

  void emitWorkloadFunc(func::FuncOp funcOp) {
    out << "inline\n";
    emitFunctionType(funcOp);
    emitWorkloadFunctionBody(funcOp);
    out << "\n";
  }

  void emitEntryFunc(func::FuncOp funcOp) {
    out << "// TODO: Emit entry " << funcOp.getName() << "\n";
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
