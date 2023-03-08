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

def GenericOp : TBLAS_Op<"generic", [NoMemoryEffect]> {
  let summary = "Generic contraction of two tensor mul.";
  let description = "None";

  let arguments = (ins F64Tensor : $input1,
                       F64Tensor : $input2,
                       I64ArrayAttr : $batch_axes1,
                       I64ArrayAttr : $batch_axes2,
                       I64ArrayAttr : $rest_axes1,
                       I64ArrayAttr : $rest_axes2,
                       I64ArrayAttr : $contract_axes1,
                       I64ArrayAttr : $contract_axes2);
  let results = (outs F64Tensor : $result);
  let builders = [OpBuilder<(ins "Value" : $input1,
                                 "Value" : $input2,
                                 "AxesTy&" : $batch_axes1,
                                 "AxesTy&" : $batch_axes2,
                                 "AxesTy&" : $rest_axes1,
                                 "AxesTy&" : $rest_axes2,
                                 "AxesTy&" : $contract_axes1,
                                 "AxesTy&" : $contract_axes2)>];
}

def BatchMatMulOp : TBLAS_Op<"batch.matmul", [NoMemoryEffect]> {
  let summary = "Batch matrix multiplication.";
  let description = "None";

  let arguments = (ins F64Tensor : $input1,
                       F64Tensor : $input2,
                       I64ArrayAttr : $batch_axes1,
                       I64ArrayAttr : $batch_axes2,
                       I64ArrayAttr : $rest_axes1,
                       I64ArrayAttr : $rest_axes2,
                       I64ArrayAttr : $contract_axes1,
                       I64ArrayAttr : $contract_axes2);
  let results = (outs F64Tensor : $result);
  let builders = [OpBuilder<(ins "Value" : $input1,
                                 "Value" : $input2,
                                 "AxesTy&" : $batch_axes1,
                                 "AxesTy&" : $batch_axes2,
                                 "AxesTy&" : $rest_axes1,
                                 "AxesTy&" : $rest_axes2,
                                 "AxesTy&" : $contract_axes1,
                                 "AxesTy&" : $contract_axes2)>];
}

#endif
