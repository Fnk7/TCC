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
  let name = "tc";
  let cppNamespace = "::mlir::contraction";
  let dependentDialects = ["func::FuncDialect", "arith::ArithDialect"];
}

class Contraction_Op<string mnemonic, list<Trait> traits = []>
    : Op<Contraction_Dialect, mnemonic, traits>;

def AddOp : Contraction_Op<"add", [NoMemoryEffect]> {
  let arguments = (ins Variadic<F64Tensor>:$inputs, IndexListArrayAttr:$axes_maps, BoolArrayAttr:$signs);
  let results = (outs F64Tensor : $result);
  let hasCanonicalizer = 1;
}

def MulOp : Contraction_Op<"mul", [NoMemoryEffect]> {
  let arguments = (ins Variadic<F64Tensor>:$inputs, IndexListArrayAttr:$axes_maps);
  let results = (outs F64Tensor : $result);
}

def NegOp : Contraction_Op<"neg", [NoMemoryEffect]> {
  let arguments = (ins F64Tensor : $input);
  let results = (outs F64Tensor : $result);
}

def ContractOp : Contraction_Op<"contract", [NoMemoryEffect]> {
  let arguments = (ins F64Tensor : $input, I64ArrayAttr : $axes);
  let results = (outs F64Tensor : $result);
}

def PermuteOp : Contraction_Op<"permute", [NoMemoryEffect]> {
  let arguments = (ins F64Tensor : $input, I64ArrayAttr : $axes_map);
  let results = (outs F64Tensor : $result);
  let hasCanonicalizer = 1;
}

def SliceOp : Contraction_Op<"slice", [NoMemoryEffect]> {
  let arguments = (ins F64Tensor : $input, Variadic<Index> : $indices, I64ArrayAttr : $axes);
  let results = (outs F64Tensor : $output);
}

def DiagonalOp : Contraction_Op<"diagonal", [NoMemoryEffect]> {
  let arguments = (ins F64Tensor : $input, IndexListArrayAttr : $axes_pairs);
  let results = (outs F64Tensor : $output);
}

def ConstantOp : Contraction_Op<"constant", [NoMemoryEffect]> {
  let arguments = (ins F64Attr : $value);
  let results = (outs F64Tensor : $output);
}

#endif
