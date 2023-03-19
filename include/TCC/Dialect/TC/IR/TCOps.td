#ifndef TCC_DIALECT_TC_IR_TCOPS_TD
#define TCC_DIALECT_TC_IR_TCOPS_TD

include "mlir/IR/OpBase.td"
// For NoMemoryEffect
include "mlir/Interfaces/SideEffectInterfaces.td"

def TC_Dialect : Dialect {
  let name = "tc";
  let summary = "Tensor Contraction Dialect";
  let cppNamespace = "::mlir::tc";
  let dependentDialects = ["func::FuncDialect"];
}

class TC_Op<string mnemonic, list<Trait> traits = []>
    : Op<TC_Dialect, mnemonic, traits # [NoMemoryEffect]>;

def MulOp : TC_Op<"mul", []> {
  let summary = "Tensor expanded elementwise multiplication.";
  let description = [{}];

  let arguments = (ins Variadic<F64Tensor>:$inputs, IndexListArrayAttr:$axes_maps);
  let results = (outs F64Tensor : $result);

  let builders = [OpBuilder<(ins "ValueRange" : $inputs,
                                 "AxesMapsTy&" : $axesMaps)>];
  let hasCanonicalizer = 1;
}

def AddOp : TC_Op<"add", [NoMemoryEffect]> {
  let summary = "Tensor elementwise addition, with signs.";
  let description = [{
    // TODO  
  }];

  let arguments = (ins Variadic<F64Tensor>:$inputs, 
                       IndexListArrayAttr:$axes_maps,
                       BoolArrayAttr:$signs);
  let results = (outs F64Tensor : $result);

  let builders = [
    OpBuilder<(ins "ValueRange" : $inputs,
                   "AxesMapsTy&" : $axesMaps,
                   "SignsTy&" : $signs)>,
    OpBuilder<(ins "ValueRange" : $inputs,
                   "AxesMapsTy&" : $axesMaps,
                   "ArrayAttr" : $signs)>,
    OpBuilder<(ins "ValueRange" : $inputs,
                   "AxesMapsTy&" : $axesMaps)>
  ];
  let hasCanonicalizer = 1;
}

def ContractOp : TC_Op<"contract", [NoMemoryEffect]> {
  let summary = "Tensor contraction.";
  let description = "TODO";

  let arguments = (ins F64Tensor : $input, I64ArrayAttr : $axes);
  let results = (outs F64Tensor : $result);

  let builders = [OpBuilder<(ins "Value" : $input, "AxesTy&" : $axes)>];
  let hasCanonicalizer = 1;
}

def PermuteOp : TC_Op<"permute", [NoMemoryEffect]> {
  let summary = "Tensor dimensions' permuting.";
  let description = "TODO";

  let arguments = (ins F64Tensor : $input, I64ArrayAttr : $axes_map);
  let results = (outs F64Tensor : $result);

  let builders = [OpBuilder<(ins "Value" : $input, "AxesMapTy&" : $axesMap)>];
  let hasCanonicalizer = 1;
}

def ConstantOp : TC_Op<"constant", [NoMemoryEffect]> {
  let summary = "Constant 0 dimension tensor.";
  let description = "TODO";

  let arguments = (ins F64Attr : $value);
  let results = (outs F64Tensor : $output);

  let builders = [OpBuilder<(ins "double" : $value)>];
  // let hasCanonicalizer = 1;
}

def NegOp : TC_Op<"neg", [NoMemoryEffect]> {
  let summary = "A negative result of a tensor.";
  let description = "TODO";

  let arguments = (ins F64Tensor : $input);
  let results = (outs F64Tensor : $result);

  let builders = [OpBuilder<(ins "Value" : $input)>];
  let hasCanonicalizer = 1;
}

def SliceOp : TC_Op<"slice", [NoMemoryEffect]> {
  let summary = "A slice of a tensor.";
  let description = "TODO";

  let arguments = (ins F64Tensor : $input, Variadic<Index> : $indices, I64ArrayAttr : $axes);
  let results = (outs F64Tensor : $result);

  let builders = [OpBuilder<(ins "Value"
                             : $input, "ValueRange"
                             : $indices, "AxesTy&"
                             : $axes)>];
  // let hasCanonicalizer = 1;
}

def DiagonalOp : TC_Op<"diagonal", [NoMemoryEffect]> {
  let summary = "Diagonal values of two axis pairs of tensor.";
  let description = "TODO";

  let arguments = (ins F64Tensor
                   : $input, I64ArrayAttr
                   : $axes_map, IndexListArrayAttr
                   : $axes_pairs);
  let results = (outs F64Tensor : $result);

  let builders = [
    OpBuilder<(ins "Value"
               : $input, "AxesMapTy&"
               : $axes)>,
    OpBuilder<(ins "Value"
               : $input, "AxesPairsTy&"
               : $axesPairs)>
  ];
}

#endif
