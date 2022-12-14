#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"

#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/Transforms/RequestCWrappers.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "TCC/Dialect/ContractionOps.h"
#include "TCC/MLIRGen.h"
#include "TCC/Parser.h"
#include "TCC/Passes.h"

using namespace llvm;

static cl::opt<std::string> InputFileName(cl::Positional,
                                          cl::desc("<Input contraction file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

namespace {
enum Action {
  DumpAST,
  DumpMLIR,
  DumpMLIRAffine,
  DumpMLIRLLVM,
  DumpLLVMIR,
};
} // namespace
static cl::opt<enum Action>
    Emit("emit", cl::desc("Select the kind of output desired"),
         cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
         cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
         cl::values(clEnumValN(DumpMLIRAffine, "mlir-affine",
                               "output the MLIR dump after affine lowering")),
         cl::values(clEnumValN(DumpMLIRLLVM, "mlir-llvm",
                               "output the MLIR dump after llvm lowering")),
         cl::values(clEnumValN(DumpLLVMIR, "llvm", "output the LLVM IR dump")));

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "TCC compiler\n");
  auto CF = contraction::parseInputFile(InputFileName);
  if (!CF)
    return 1;
  if (Emit == DumpAST) {
    CF->dump();
    return 0;
  }
  mlir::MLIRContext Ctx;
  Ctx.getOrLoadDialect<mlir::scf::SCFDialect>();
  Ctx.getOrLoadDialect<mlir::func::FuncDialect>();
  Ctx.getOrLoadDialect<mlir::memref::MemRefDialect>();
  Ctx.getOrLoadDialect<mlir::contraction::ContractionDialect>();
  mlir::PassManager PM(&Ctx);
  auto M = mlir::contraction::MLIRGen(Ctx, *CF);
  if (Emit == DumpMLIR) {
    mlir::OpPassManager &FuncPM = PM.nest<mlir::func::FuncOp>();
    FuncPM.addPass(mlir::createCanonicalizerPass());
    FuncPM.addPass(mlir::createCSEPass());
    mlir::failed(PM.run(*M));
    M->dump();
    return 0;
  }
  PM.addPass(mlir::contraction::createLowerToAffinePass());
  mlir::OpPassManager &FuncPM = PM.nest<mlir::func::FuncOp>();
  FuncPM.addPass(mlir::contraction::createReturnToArgumentPass());
  FuncPM.addPass(mlir::createCanonicalizerPass());
  FuncPM.addPass(mlir::createCSEPass());
  FuncPM.addPass(mlir::createLoopFusionPass());
  FuncPM.addPass(mlir::createAffineScalarReplacementPass());
  FuncPM.addPass(mlir::LLVM::createRequestCWrappersPass());
  if (Emit == DumpMLIRLLVM || Emit == DumpLLVMIR) {
    PM.addPass(mlir::contraction::createLowerToLLVMPass());
  }
  mlir::failed(PM.run(*M));
  if (Emit == DumpMLIRAffine || Emit == DumpMLIRLLVM) {
    M->dump();
    return 0;
  }
  mlir::registerLLVMDialectTranslation(*M->getContext());
  llvm::LLVMContext LLVMCtx;
  auto LLVMIR = mlir::translateModuleToLLVMIR(M.get(), LLVMCtx);
  errs() << *LLVMIR << "\n";
  return 0;
}
