#ifndef TCC_DIALECT_TBLAS_TRANSFORMS_PASSES_TD
#define TCC_DIALECT_TBLAS_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def Degeneralize : Pass<"tblas-degeneralize", "func::FuncOp"> {
  let constructor = "mlir::tc::createTBLASDegeneralizePass()";
  let dependentDialects = ["func::FuncDialect"];
}

def SpecialGemm : Pass<"tblas-special-gemm", "func::FuncOp"> {
  let constructor = "mlir::tc::createTBLASSpecialGemmPass()";
  let dependentDialects = ["func::FuncDialect"];
}

#endif
