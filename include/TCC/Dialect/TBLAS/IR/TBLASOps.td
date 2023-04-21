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
}

def BatchGemmOp : TBLAS_Op<"batch.gemm", [NoMemoryEffect]> {
  let summary = "Batch general matrix multiplication.";
  let description = "None";

  let arguments = (ins F64Tensor : $lhs,
                       F64Tensor : $rhs,
                       I64Attr : $num_batch,
                       BoolAttr : $lhs_trans,
                       BoolAttr : $rhs_trans,
                       I64Attr : $num_M,
                       I64Attr : $num_N,
                       I64Attr : $num_K);
  let results = (outs F64Tensor : $result);
}

def BatchGemvOp : TBLAS_Op<"batch.gemv", [NoMemoryEffect]> {
  let summary = "Batch general matrix vector multiplication.";
  let description = "None";

  let arguments = (ins F64Tensor : $matrix,
                       F64Tensor : $vector,
                       I64Attr : $num_batch,
                       BoolAttr : $trans,
                       I64Attr : $num_M,
                       I64Attr : $num_N);
  let results = (outs F64Tensor : $result);
}

def BatchDotOp : TBLAS_Op<"batch.dot", [NoMemoryEffect]> {
  let summary = "Batch general vector multiplication.";
  let description = "None";

  let arguments = (ins F64Tensor : $lhs,
                       F64Tensor : $rhs,
                       I64Attr : $num_batch);
  let results = (outs F64Tensor : $result);
}

def BatchSyr2kOp : TBLAS_Op<"batch.syr2k", [NoMemoryEffect]> {
  let summary = "Batch matrix multiplication and transpose add.";
  let description = "None";

  let arguments = (ins F64Tensor : $lhs,
                       F64Tensor : $rhs,
                       I64Attr : $num_batch,
                       BoolAttr : $trans,
                       I64Attr : $num_N,
                       I64Attr : $num_K);
  let results = (outs F64Tensor : $result);
}

def BatchSyrkOp : TBLAS_Op<"batch.syrk", [NoMemoryEffect]> {
  let summary = "Batch matrix and self transpose multiplication.";
  let description = "None";

  let arguments = (ins F64Tensor : $input,
                       I64Attr : $num_batch,
                       BoolAttr : $trans,
                       I64Attr : $num_N,
                       I64Attr : $num_K);
  let results = (outs F64Tensor : $result);
}

def BatchSyrkxOp : TBLAS_Op<"batch.syrkx", [NoMemoryEffect]> {
  let summary = "Batch matrix multiplication, result is symmetry.";
  let description = "None";

  let arguments = (ins F64Tensor : $lhs,
                       F64Tensor : $rhs,
                       I64Attr : $num_batch,
                       BoolAttr : $lhs_trans,
                       BoolAttr : $rhs_trans,
                       I64Attr : $num_N,
                       I64Attr : $num_K);
  let results = (outs F64Tensor : $result);
}

def BatchTrmmOp : TBLAS_Op<"batch.trmm", [NoMemoryEffect]> {
  let summary = "Batch triangular matrix and matrix multiplication.";
  let description = "None";

  let arguments = (ins F64Tensor : $input,
                       I64Attr : $num_batch,
                       BoolAttr : $lhs_trans,
                       BoolAttr : $rhs_trans,
                       I64Attr : $num_N,
                       I64Attr : $num_K);
  let results = (outs F64Tensor : $result);
}

#endif
