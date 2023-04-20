#include "TCC/CodeGen.h"

#include "mlir/Dialect/Affine/Passes.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

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

#include "TCC/Dialect/TBLAS/IR/TBLASOps.h"
#include "TCC/Dialect/TBLAS/Transforms/Passes.h"
#include "TCC/Dialect/TC/IR/TCOps.h"
#include "TCC/Dialect/TC/Transforms/Passes.h"
#include "TCC/MLIRGen.h"
#include "TCC/Parser.h"
#include "TCC/Passes.h"

#include <zstd.h>

using namespace llvm;

static cl::opt<std::string> inputFileName(cl::Positional,
                                          cl::desc("<Input contraction file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

static cl::opt<bool> TCNoOpt("tc-no-opt", cl::desc("TC Dialect Optimization"),
                             cl::init(false));

static cl::opt<bool> LinalgFuse("linalg-fuse",
                                cl::desc("Fuse linalg generic parallel loop."),
                                cl::init(false));
static cl::opt<bool> AddBatch("add-batch",
                              cl::desc("Add batch for calculation."),
                              cl::init(false));

static cl::opt<bool> LowerLinalg("lower-linalg",
                                 cl::desc("Lower linalg dialect"),
                                 cl::init(false));

namespace {
enum Action {
  DumpAST,
  DumpMLIR,
  DumpMLIRLINALG,
  DumpMLIRAffine,
  DumpMLIRLLVM,
  DumpLLVMIR,
  DumpMLIRBLAS,
  EmitScopC,
  EmitTVM,
  EmitRELAY,
};
} // namespace
static cl::opt<enum Action>
    Emit("emit", cl::desc("Select the kind of output desired"),
         cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
         cl::values(clEnumValN(EmitScopC, "scop-c", "emit scop c")),
         cl::values(clEnumValN(EmitTVM, "tvm", "emit tvm tensor expressions")),
         cl::values(clEnumValN(EmitRELAY, "relay", "emit tvm relay")),
         cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
         cl::values(clEnumValN(DumpMLIRLINALG, "mlir-linalg",
                               "output the MLIR dump after linalg lowering")),
         cl::values(clEnumValN(DumpMLIRAffine, "mlir-affine",
                               "output the MLIR dump after affine lowering")),
         cl::values(clEnumValN(DumpMLIRLLVM, "mlir-llvm",
                               "output the MLIR dump after llvm lowering")),
         cl::values(clEnumValN(DumpMLIRBLAS, "mlir-blas",
                               "output the MLIR dump after llvm lowering")),
         cl::values(clEnumValN(DumpLLVMIR, "llvm", "output the LLVM IR dump")));

namespace {
void addTCOptimizePipeline(mlir::PassManager &PM) {
  mlir::OpPassManager &funcPM = PM.nest<mlir::func::FuncOp>();
  // TOcFIX: only one call
  funcPM.addPass(mlir::createCanonicalizerPass());
  funcPM.addPass(mlir::createCSEPass());
  funcPM.addPass(mlir::createCanonicalizerPass());
  funcPM.addPass(mlir::createCSEPass());
  PM.addPass(mlir::tc::createTCOptimizePass());
  funcPM.addPass(mlir::createCSEPass());
  if (AddBatch) {
    PM.addPass(mlir::tc::createAddBatchPass());
  }
}

void addConvertToLinalgPipeline(mlir::PassManager &PM,
                                bool eraseEncoding = false) {
  PM.addPass(mlir::tc::createConvertTCToLinalgPass(eraseEncoding));
  PM.addPass(mlir::createCSEPass());
  PM.addPass(mlir::createCanonicalizerPass());
}

void addLinalgFusePass(mlir::PassManager &PM) {
  if (LinalgFuse) {
    PM.addPass(mlir::createLinalgElementwiseOpFusionPass());
    PM.addPass(mlir::createCSEPass());
  }
}
} // namespace

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "Tensor Contraction MLIR Compiler\n");
  auto CF = tc::parseInputFile(inputFileName);
  if (!CF)
    return 1;
  if (Emit == DumpAST) {
    CF->dump();
    return 0;
  }
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::tc::TCDialect>();
  context.getOrLoadDialect<mlir::tc::TBLASDialect>();
  mlir::PassManager PM(&context);
  auto M = mlir::tc::MLIRGen(context, *CF);
  if (Emit == DumpMLIR) {
    if (!TCNoOpt) {
      addTCOptimizePipeline(PM);
      mlir::failed(PM.run(*M));
    }
    SmallString<32> outputFileName(inputFileName);
    sys::path::replace_extension(outputFileName, ".mlir");
    std::error_code EC;
    raw_fd_ostream dest(outputFileName, EC, sys::fs::OF_Text);
    if (EC) {
      errs() << "Could not open file: " << EC.message();
      return 1;
    }
    M->print(dest);
    return 0;
  }

  if (Emit == EmitScopC) {
    addTCOptimizePipeline(PM);
    addConvertToLinalgPipeline(PM);
    addLinalgFusePass(PM);
    mlir::failed(PM.run(*M));
    SmallString<32> outputFileName(inputFileName);
    sys::path::replace_extension(outputFileName, ".scop.c");
    std::error_code EC;
    raw_fd_ostream dest(outputFileName, EC, sys::fs::OF_Text);
    if (EC) {
      errs() << "Could not open file: " << EC.message();
      return 1;
    }
    mlir::tc::codegen::emitLinalgToC(M.get(), dest);
    return 0;
  }

  if (Emit == EmitTVM || Emit == EmitRELAY) {
    addTCOptimizePipeline(PM);
    mlir::failed(PM.run(*M));
    SmallString<32> outputFileName(inputFileName);
    sys::path::replace_extension(outputFileName,
                                 Emit == EmitTVM ? ".tvm.py" : ".tvm.relay.py");
    std::error_code EC;
    raw_fd_ostream dest(outputFileName, EC, sys::fs::OF_Text);
    if (EC) {
      errs() << "Could not open file: " << EC.message();
      return 1;
    }
    if (Emit == EmitTVM)
      mlir::tc::codegen::emitTCToTVM(M.get(), dest);
    else
      mlir::tc::codegen::emitTCToTVMRelay(M.get(), dest);
    return 0;
  }

  if (Emit == DumpMLIRBLAS) {
    addTCOptimizePipeline(PM);
    mlir::OpPassManager &FuncPM2 = PM.nest<mlir::func::FuncOp>();
    FuncPM2.addPass(mlir::tc::createConvertTCToTBLASPass());
    PM.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
    PM.addNestedPass<mlir::func::FuncOp>(mlir::tc::createTCDCEPass());
    FuncPM2.addPass(mlir::tc::createTBLASDegeneralizePass());
    PM.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
    PM.addNestedPass<mlir::func::FuncOp>(mlir::tc::createTCDCEPass());
    FuncPM2.addPass(mlir::tc::createTBLASSpecialGemmPass());
    PM.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
    PM.addNestedPass<mlir::func::FuncOp>(mlir::tc::createTCDCEPass());
    PM.addPass(mlir::tc::createSplitWorkloadPass());
    PM.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
    PM.addNestedPass<mlir::func::FuncOp>(mlir::tc::createTCDCEPass());
    mlir::failed(PM.run(*M));
    SmallString<32> outputFileName(inputFileName);
    sys::path::replace_extension(outputFileName, ".blas.mlir");
    std::error_code EC;
    raw_fd_ostream dest(outputFileName, EC, sys::fs::OF_Text);
    if (EC) {
      errs() << "Could not open file: " << EC.message();
      return 1;
    }
    M->print(dest);
    return 0;
  }

  if (Emit == DumpMLIRLINALG) {
    addTCOptimizePipeline(PM);
    addConvertToLinalgPipeline(PM, true);
    addLinalgFusePass(PM);
    if (LowerLinalg) {
      PM.addPass(mlir::func::createFuncBufferizePass());
      mlir::OpPassManager &FuncPM3 = PM.nest<mlir::func::FuncOp>();
      FuncPM3.addPass(mlir::createLinalgBufferizePass());
      PM.addPass(mlir::createCSEPass());
      PM.addPass(mlir::tc::createReturnToArgumentPass());
      mlir::OpPassManager &FuncPM4 = PM.nest<mlir::func::FuncOp>();
      FuncPM4.addPass(mlir::createConvertLinalgToLoopsPass());
      PM.addPass(mlir::createBufferizationToMemRefPass());
    }
    mlir::failed(PM.run(*M));
    M->print(llvm::outs());
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
