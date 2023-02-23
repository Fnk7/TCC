#include "mlir/Dialect/Affine/Passes.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"

#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/RequestCWrappers.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "TCC/Dialect/TC/IR/TCOps.h"
#include "TCC/Dialect/TC/Transforms/Passes.h"
#include "TCC/MLIRGen.h"
#include "TCC/Parser.h"
#include "TCC/Passes.h"

using namespace llvm;

static cl::opt<std::string> InputFileName(cl::Positional,
                                          cl::desc("<Input contraction file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

static cl::opt<bool> CTCSE("opt-cse", cl::desc("Contraction Dialect CSE Opt"),
                           cl::init(false));

namespace {
enum Action {
  DumpAST,
  DumpMLIR,
  DumpMLIRLINALG,
  DumpMLIRAffine,
  DumpMLIRLLVM,
  DumpLLVMIR,
};
} // namespace
static cl::opt<enum Action>
    Emit("emit", cl::desc("Select the kind of output desired"),
         cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
         cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
         cl::values(clEnumValN(DumpMLIRLINALG, "mlir-linalg",
                               "output the MLIR dump after linalg lowering")),
         cl::values(clEnumValN(DumpMLIRAffine, "mlir-affine",
                               "output the MLIR dump after affine lowering")),
         cl::values(clEnumValN(DumpMLIRLLVM, "mlir-llvm",
                               "output the MLIR dump after llvm lowering")),
         cl::values(clEnumValN(DumpLLVMIR, "llvm", "output the LLVM IR dump")));

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "TCC compiler\n");
  auto CF = tc::parseInputFile(InputFileName);
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
  Ctx.getOrLoadDialect<mlir::tc::TCDialect>();
  Ctx.getOrLoadDialect<mlir::bufferization::BufferizationDialect>();
  mlir::PassManager PM(&Ctx);
  auto M = mlir::tc::MLIRGen(Ctx, *CF);
  if (Emit == DumpMLIR) {
    if (CTCSE) {
      mlir::OpPassManager &FuncPM = PM.nest<mlir::func::FuncOp>();
      // TOcFIX: only one call
      FuncPM.addPass(mlir::createCanonicalizerPass());
      FuncPM.addPass(mlir::createCSEPass());
      FuncPM.addPass(mlir::createCanonicalizerPass());
      FuncPM.addPass(mlir::createCSEPass());
      // FuncPM.addPass(mlir::tc::createMergeContractPass());
      // FuncPM.addPass(mlir::createCSEPass());
      // FuncPM.addPass(mlir::tc::createSplitContractPass());
      // FuncPM.addPass(mlir::createCSEPass());
      // FuncPM.addPass(mlir::tc::createGroupAddWithSameInputPass());
      // FuncPM.addPass(mlir::createCSEPass());
      // FuncPM.addPass(mlir::tc::createPrintSymmetryPass());
      // FuncPM.addPass(mlir::tc::createSplitContractPass());
      // FuncPM.addPass(mlir::createCSEPass());
      // FuncPM.addPass(mlir::tc::createMergeContractPass());
      // FuncPM.addPass(mlir::createCSEPass());
      PM.addPass(mlir::tc::createTCOptimizePass());
      mlir::OpPassManager &FuncPM2 = PM.nest<mlir::func::FuncOp>();
      FuncPM2.addPass(mlir::LLVM::createRequestCWrappersPass());
      mlir::failed(PM.run(*M));
    }
    M->dump();
    return 0;
  }
  if (Emit == DumpMLIRLINALG) {
    mlir::OpPassManager &FuncPM = PM.nest<mlir::func::FuncOp>();
    FuncPM.addPass(mlir::createCanonicalizerPass());
    FuncPM.addPass(mlir::createCSEPass());
    FuncPM.addPass(mlir::createCanonicalizerPass());
    FuncPM.addPass(mlir::createCSEPass());
    PM.addPass(mlir::tc::createTCOptimizePass());
    mlir::OpPassManager &FuncPM2 = PM.nest<mlir::func::FuncOp>();
    FuncPM2.addPass(mlir::LLVM::createRequestCWrappersPass());
    PM.addPass(mlir::tc::createConvertTCToLinalgPass());
    PM.addPass(mlir::createCSEPass());
    PM.addPass(mlir::createLinalgElementwiseOpFusionPass());
    PM.addPass(mlir::func::createFuncBufferizePass());
    mlir::OpPassManager &FuncPM3 = PM.nest<mlir::func::FuncOp>();
    FuncPM3.addPass(mlir::createLinalgBufferizePass());
    PM.addPass(mlir::createCSEPass());
    PM.addPass(mlir::tc::createReturnToArgumentPass());
    mlir::OpPassManager &FuncPM4 = PM.nest<mlir::func::FuncOp>();
    FuncPM4.addPass(mlir::createConvertLinalgToLoopsPass());
    PM.addPass(mlir::createBufferizationToMemRefPass());
    mlir::failed(PM.run(*M));
    M->dump();
    return 0;
  }
  PM.addPass(mlir::tc::createLowerToAffinePass());
  mlir::OpPassManager &FuncPM = PM.nest<mlir::func::FuncOp>();
  FuncPM.addPass(mlir::tc::createReturnToArgumentPass());
  FuncPM.addPass(mlir::createCanonicalizerPass());
  FuncPM.addPass(mlir::createCSEPass());
  FuncPM.addPass(mlir::createLoopFusionPass());
  FuncPM.addPass(mlir::createAffineScalarReplacementPass());
  // FuncPM.addPass(mlir::LLVM::createRequestCWrappersPass());
  if (Emit == DumpMLIRLLVM || Emit == DumpLLVMIR) {
    PM.addPass(mlir::tc::createLowerToLLVMPass());
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
