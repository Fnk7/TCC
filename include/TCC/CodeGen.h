#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/Support/raw_ostream.h"

namespace mlir::tc::codegen {
class CodeGenBase {
protected:
  llvm::raw_ostream &out;
  CodeGenBase(llvm::raw_ostream &out) : out(out){};
};

void emitLinalgToC(ModuleOp op, llvm::raw_ostream &out = llvm::errs());
void emitTCToTVM(ModuleOp op, llvm::raw_ostream &out = llvm::errs());
void emitTCToTVMRelay(ModuleOp op, llvm::raw_ostream &out = llvm::errs());
void emitTBLASToOpenCL(ModuleOp op, llvm::raw_ostream &out,
                       bool workgroup = true);
void emitBLASWithLinalgToC(ModuleOp op, llvm::raw_ostream &out = llvm::errs());
} // namespace mlir::tc::codegen
