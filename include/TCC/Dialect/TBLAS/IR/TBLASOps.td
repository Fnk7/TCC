#ifndef TCC_DIALECT_TBLAS_IR_TBLASOPS_TD
#define TCC_DIALECT_TBLAS_IR_TBLASOPS_TD

include "mlir/IR/OpBase.td"
include "TCC/Dialect/TC/IR/TCAttributes.td"
// For NoMemoryEffect
include "mlir/Interfaces/SideEffectInterfaces.td"

def TBLAS_Dialect : Dialect {
  let name = "tblas";
  let summary = "Tensor BLAS, with batch.";
  let cppNamespace = "::mlir::tc";
  let dependentDialects = ["::mlir::tc::TCDialect"];
}

class TBLAS_Op<string mnemonic, list<Trait> traits = []>
    : Op<TBLAS_Dialect, mnemonic, traits>;

def GenericOp : TBLAS_Op<"generic", [NoMemoryEffect]> {
  let summary = "Generic contraction of two tensor mul.";
  let description = "None";

  let arguments = (ins F64Tensor : $lhs,
                       F64Tensor : $rhs,
                       DenseI64ArrayAttr : $lhs_batch,
                       DenseI64ArrayAttr : $rhs_batch,
                       DenseI64ArrayAttr : $lhs_rest,
                       DenseI64ArrayAttr : $rhs_rest,
                       DenseI64ArrayAttr : $lhs_contract,
                       DenseI64ArrayAttr : $rhs_contract);
  let results = (outs F64Tensor : $result);
  // let builders = [OpBuilder<(ins "Type" : $result,
  //                                "Value" : $lhs,
  //                                "Value" : $rhs,
  //                                "AxesRefTy" : $lhs_batch,
  //                                "AxesRefTy" : $rhs_batch,
  //                                "AxesRefTy" : $lhs_rest,
  //                                "AxesRefTy" : $rhs_rest,
  //                                "AxesRefTy" : $lhs_contract,
  //                                "AxesRefTy" : $rhs_contract)>];
}

def BatchMatMulOp : TBLAS_Op<"batch.matmul", [NoMemoryEffect]> {
  let summary = "Batch matrix multiplication.";
  let description = "None";

  let arguments = (ins F64Tensor : $lhs,
                       F64Tensor : $rhs,
                       I64ArrayAttr : $lhs_batch,
                       I64ArrayAttr : $rhs_batch,
                       I64ArrayAttr : $lhs_rest,
                       I64ArrayAttr : $rhs_rest,
                       I64ArrayAttr : $lhs_contract,
                       I64ArrayAttr : $rhs_contract);
  let results = (outs F64Tensor : $result);
  let builders = [OpBuilder<(ins "Value" : $lhs,
                                 "Value" : $rhs,
                                 "AxesTy&" : $lhs_batch,
                                 "AxesTy&" : $rhs_batch,
                                 "AxesTy&" : $lhs_rest,
                                 "AxesTy&" : $rhs_rest,
                                 "AxesTy&" : $lhs_contract,
                                 "AxesTy&" : $rhs_contract)>];
}

#endif
