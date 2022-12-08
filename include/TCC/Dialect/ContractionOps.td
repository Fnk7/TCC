#ifndef CONTRACTION_DIALECT_CONTRACTIONOPS
#define CONTRACTION_DIALECT_CONTRACTIONOPS

include "mlir/IR/OpBase.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/ParallelCombiningOpInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"

def Contraction_Dialect : Dialect {
  let name = "contraction";
  let cppNamespace = "::mlir::contraction";
  let dependentDialects = ["func::FuncDialect"];
}

class Contraction_Op<string mnemonic, list<Trait> traits = []>
    : Op<Contraction_Dialect, mnemonic, traits>;

def AddOp : Contraction_Op<"add", []> {
  let arguments = (ins Variadic<F64Tensor>:$tensors, IndexListArrayAttr:$axes_maps);
  let results = (outs F64Tensor : $result);
}

def MulOp : Contraction_Op<"mul", []> {
  let arguments = (ins Variadic<F64Tensor>:$tensors, IndexListArrayAttr:$axes_maps);
  let results = (outs F64Tensor : $result);
}

def ContractOp : Contraction_Op<"contract", []> {
  let arguments = (ins F64Tensor : $tensor, I64ArrayAttr : $axes);
  let results = (outs F64Tensor : $result);
}

def PermuteOp : Contraction_Op<"permute", []> {
  let arguments = (ins F64Tensor : $tensor, I64ArrayAttr : $axes_map);
  let results = (outs F64Tensor : $result);
}

def SliceOp : Contraction_Op<"slice", []> {
  let arguments = (ins F64Tensor : $input, Variadic<Index> : $indices, I64ArrayAttr : $axes);
  let results = (outs F64Tensor : $output);
}

def DiagonalOp : Contraction_Op<"diagonal", []> {
  let arguments = (ins F64Tensor : $input, I64ArrayAttr : $axes);
  let results = (outs F64Tensor : $output);
}

#endif
