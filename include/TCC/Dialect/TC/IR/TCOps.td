#ifndef TCC_DIALECT_TC_IR_TCOPS_TD
#define TCC_DIALECT_TC_IR_TCOPS_TD

include "./TCBase.td"
include "./TCAttributes.td"
// For NoMemoryEffect
include "mlir/Interfaces/SideEffectInterfaces.td"
// For ReturnLike
include "mlir/Interfaces/ControlFlowInterfaces.td"

class TC_Op<string mnemonic, list<Trait> traits = []>
    : Op<TC_Dialect, mnemonic, traits #[NoMemoryEffect]>;

def MulOp : TC_Op<"mul", []> {
  let summary = "Tensor expanded elementwise multiplication.";
  let description = [{}];

  let arguments = (ins Variadic<F64Tensor> : $inputs,
                       AxesMapsAttr : $axes_maps);
  let results = (outs F64Tensor : $result);

  let builders = [OpBuilder<(ins "ValueRange" : $inputs,
                                 "AxesMapsRefTy" : $axesMaps)>];
  let extraClassDeclaration = [{
    /// Helper function to get axes_maps as SmallVector<SmallVector<AxisTy>>.
    AxesMapsTy getAxesMapsArray();
  }];
  let hasCanonicalizer = 1;
  let assemblyFormat = "(`(` $inputs^ `)`)? "
                       "`{` `axesmaps` `=` $axes_maps `}` `:` attr-dict ` ` "
                       "`(`type($inputs)`)` `->` type($result)";
}

def AddOp : TC_Op<"add", [NoMemoryEffect]> {
  let summary = "Tensor elementwise addition, with signs.";
  let description = [{
      // TODO
  }];

  let arguments = (ins Variadic<F64Tensor> : $inputs,
                       AxesMapsAttr : $axes_maps,
                       SignsAttr : $signs);
  let results = (outs F64Tensor : $result);

  let builders = [
    OpBuilder<(ins "ValueRange" : $inputs,
                   "AxesMapsTy&" : $axesMaps,
                   "SignsTy&" : $signs)>,
    OpBuilder<(ins "ValueRange" : $inputs,
                   "AxesMapsTy&" : $axesMaps,
                   "SignsAttr" : $signs)>,
    OpBuilder<(ins "ValueRange" : $inputs,
                   "AxesMapsTy&" : $axesMaps)>
  ];
  let extraClassDeclaration = [{
    /// Helper function to get axes_maps as SmallVector<SmallVector<AxisTy>>.
    AxesMapsTy getAxesMapsArray();
    /// Helper function to get signs as SmallVector<bool>.
    SignsTy getSignsArray();
  }];
  let hasCanonicalizer = 1;
  let assemblyFormat = "(`(` $inputs^ `)`)? "
                       "`{` `axesmaps` `=` $axes_maps `,`"
                       " `signs` `=` $signs `}` `:` attr-dict ` ` "
                       "`(`type($inputs)`)` `->` type($result)";
}

def ContractOp : TC_Op<"contract", [NoMemoryEffect]> {
  let summary = "Tensor contraction.";
  let description = "TODO";

  let arguments = (ins F64Tensor : $input, AxesAttr : $axes);
  let results = (outs F64Tensor : $result);

  let builders = [OpBuilder<(ins "Value" : $input, "AxesTy&" : $axes)>];
  let extraClassDeclaration = [{
    /// Helper function to get axes as SmallVector<AxisTy>.
    AxesTy getAxesArray();
  }];
  let hasCanonicalizer = 1;
  let assemblyFormat = "`(` $input `)` "
                       "`{` `axes` `=` $axes `}` `:` attr-dict ` ` "
                       "type($input) `->` type($result)";
}

def PermuteOp : TC_Op<"permute", [NoMemoryEffect]> {
  let summary = "Tensor dimensions' permuting.";
  let description = "TODO";

  let arguments = (ins F64Tensor : $input, AxesMapsAttr : $axes_map);
  let results = (outs F64Tensor : $result);

  let builders = [OpBuilder<(ins "Value" : $input, "AxesMapTy&" : $axesMap)>];
  let extraClassDeclaration = [{
    /// Helper function to get axes_map as SmallVector<AxisTy>.
    AxesMapTy getAxesMapArray();
  }];
  let hasCanonicalizer = 1;
  let assemblyFormat = "`(` $input `)` "
                       "`{` `axesmap` `=` $axes_map `}` `:` attr-dict ` ` "
                       "type($input) `->` type($result)";
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

  let arguments = (ins F64Tensor : $input, Variadic<Index> : $indices, AxesAttr : $axes);
  let results = (outs F64Tensor : $result);

  let builders = [OpBuilder<(ins "Value" : $input,
                                 "ValueRange" : $indices,
                                 "AxesTy&" : $axes)>];
  // let hasCanonicalizer = 1;
}

def DiagonalOp : TC_Op<"diagonal", [NoMemoryEffect]> {
  let summary = "Diagonal values of two axis pairs of tensor.";
  let description = "TODO";

  let arguments = (ins F64Tensor : $input,
                       I64ArrayAttr : $axes_map,
                       IndexListArrayAttr : $axes_pairs);
  let results = (outs F64Tensor : $result);

  let builders = [
    OpBuilder<(ins "Value" : $input,
                   "AxesMapTy&" : $axes)>,
    OpBuilder<(ins "Value" : $input,
                   "AxesPairsTy&" : $axesPairs)>
  ];
}

def ReturnOp
    : Op<TC_Dialect, "return",
         [Pure, HasParent<"::mlir::func::FuncOp">, ReturnLike, Terminator]> {
  let summary = "Function return operation allow implicit type cast.";
  let description = [{}];
  let arguments = (ins Variadic<AnyType> : $operands);
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

#endif
