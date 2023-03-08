#ifndef TCC_DIALECT_TBLAS_IR_TBLASOPS_TD
#define TCC_DIALECT_TBLAS_IR_TBLASOPS_TD

include "mlir/IR/OpBase.td"
// For NoMemoryEffect
include "mlir/Interfaces/SideEffectInterfaces.td"

def TBLAS_Dialect : Dialect {
  let name = "tblas";
  let summary = "Tensor BLAS, with batch.";
  let cppNamespace = "::mlir::tblas";
  let dependentDialects = [];
}

class TBLAS_Op<string mnemonic, list<Trait> traits = []>
    : Op<TBLAS_Dialect, mnemonic, traits>;

def BatchMatMul : TBLAS_Op<"batch.matmul", [NoMemoryEffect]> {
  let summary = "Batch matrix multiplication.";
  let description = "None";

  let arguments = (ins F64Tensor:$input1, F64Tensor:$input2);
  let results = (outs F64Tensor : $result);

  let builders = [];
}

#endif
