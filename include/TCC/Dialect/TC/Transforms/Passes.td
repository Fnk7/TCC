#ifndef TCC_DIALECT_TC_TRANSFORMS_PASSES_TD
#define TCC_DIALECT_TC_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def PrintSymmetry : Pass<"print-symmetry", "func::FuncOp"> {
  let constructor = "mlir::tc::createPrintSymmetryPass()";
  let dependentDialects = ["func::FuncDialect"];
}

def MergeContract : Pass<"merge-contract", "func::FuncOp"> {
  let constructor = "mlir::tc::createMergeContractPass()";
  let dependentDialects = ["func::FuncDialect"];
}

def SplitContract : Pass<"split-contract", "func::FuncOp"> {
  let constructor = "mlir::tc::createSplitContractPass()";
  let dependentDialects = ["func::FuncDialect"];
}

def GroupAddWithSameInput : Pass<"group-add", "func::FuncOp"> {
  let constructor = "mlir::tc::createGroupAddWithSameInputPass()";
  let dependentDialects = ["func::FuncDialect"];
}

def AddContractMulTreeAnalysis : Pass<"tree-analysis", "func::FuncOp"> {
  let constructor = "mlir::tc::createAddContractMulTreeAnalysisPass()";
  let dependentDialects = ["func::FuncDialect"];
}

def MergeMul : Pass<"merge-mul", "func::FuncOp"> {
  let constructor = "mlir::tc::createMergeMulPass()";
  let dependentDialects = ["func::FuncDialect"];
}

def TCOptimize : Pass<"tc-opt", "ModuleOp"> {
  let constructor = "mlir::tc::createTCOptimizePass()";
  let dependentDialects = ["func::FuncDialect"];
}

def TCCSE : Pass<"tc-cse", "func::FuncOp"> {
  let constructor = "mlir::tc::createTCCSEPass()";
  let dependentDialects = ["func::FuncDialect"];
}

def SinkMul : Pass<"tc-sink-mul", "func::FuncOp"> {
  let constructor = "mlir::tc::createSinkMulPass()";
  let dependentDialects = ["func::FuncDialect"];
}

def LiftMul : Pass<"tc-litf-mul", "func::FuncOp"> {
  let constructor = "mlir::tc::createLiftMulPass()";
  let dependentDialects = ["func::FuncDialect"];
}

#endif
