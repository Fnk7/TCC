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

#endif
