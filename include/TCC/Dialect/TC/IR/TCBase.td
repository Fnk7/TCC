#ifndef TCC_DIALECT_TC_IR_TCBASE_TD
#define TCC_DIALECT_TC_IR_TCBASE_TD

include "mlir/IR/OpBase.td"

def TC_Dialect : Dialect {
  let name = "tc";
  let summary = "Tensor Contraction Dialect";
  let cppNamespace = "::mlir::tc";
  let dependentDialects = ["func::FuncDialect"];
}

#endif
