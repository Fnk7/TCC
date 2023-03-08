#ifndef TCC_DIALECT_TBLAS_TRANSFORMS_PASSES_TD
#define TCC_DIALECT_TBLAS_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def TBLASTOTO : Pass<"tblas-toto", "func::FuncOp"> {
  let constructor = "mlir::tc::createTBLASTOTOPass()";
  let dependentDialects = ["func::FuncDialect"];
}

#endif
