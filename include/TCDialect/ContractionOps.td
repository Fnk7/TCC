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
}

class Contraction_Op<string mnemonic, list<Trait> traits = []> :
    Op<Contraction_Dialect, mnemonic, traits>;



def AddOp : Contraction_Op<"add", []> {
    let arguments = (ins F64:$lhs, F64:$rhs);
    let results = (outs F64);
    let builders = [
        OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
    ];
}

def MulOp : Contraction_Op<"mul", []> {
    let arguments = (ins F64:$lhs, F64:$rhs);
    let results = (outs F64);
    let builders = [
        OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
    ];
}

def ForOp : Contraction_Op<"for", [
    DeclareOpInterfaceMethods<RegionBranchOpInterface>,
    SingleBlockImplicitTerminator<"contraction::YieldOp">]> {
    let arguments = (ins Variadic<Index>:$ranges);
    let results = (outs AnyTypeOf<[F64Tensor, F64MemRef]>);
    let regions = (region SizedRegion<1>:$region);
    let builders = [
        OpBuilder<(ins "ValueRange":$ranges)>
    ];
}

def ContractOp : Contraction_Op<"contraction", [
    DeclareOpInterfaceMethods<RegionBranchOpInterface>,
    SingleBlockImplicitTerminator<"contraction::YieldOp">]> {
    let arguments = (ins Variadic<Index>:$ranges);
    let results = (outs F64);
    let regions = (region SizedRegion<1>:$region);
    let builders = [
        OpBuilder<(ins "ValueRange":$ranges)>
    ];
}

def YieldOp : Contraction_Op<"yield", [Pure, ReturnLike, Terminator, ParentOneOf<["ForOp", "ContractOp"]>]> {
    let arguments = (ins F64:$result);
    let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
}

def IndexOp : Contraction_Op<"index", []> {
    let arguments = (ins AnyTypeOf<[F64Tensor, F64MemRef]>:$tensor, Variadic<Index>:$indexes);
    let results = (outs F64);
    let builders = [
        OpBuilder<(ins "Value":$tensor, "ValueRange":$indexes)>
    ];
}

#endif
