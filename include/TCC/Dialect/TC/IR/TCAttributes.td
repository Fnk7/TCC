#ifndef TCC_DIALECT_TC_IR_TCATTRIBUTE_TD
#define TCC_DIALECT_TC_IR_TCATTRIBUTE_TD

include "./TCBase.td"
include "mlir/IR/AttrTypeBase.td"

class TC_Attr<string attrName, string attrMnemonic,
                   list<Trait> traits = []>
    : AttrDef<TC_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

def AxesParameter : ArrayRefParameter<"::mlir::tc::AxisTy", "Axes">;
def AxesMapsParameter : ArrayRefParameter<"::mlir::tc::AxesMapTy", "Axes Maps">;

def AxesAttr : TC_Attr<"Axes", "axes"> {
  let parameters = (ins AxesParameter:$axes);
  let hasCustomAssemblyFormat = 1;
  let returnType = "::mlir::tc::AxesRefTy";
  let convertFromStorage = "$_self.getAxes()";
}

def AxesMapsAttr : TC_Attr<"AxesMaps", "axesmaps"> {
  let parameters = (ins AxesMapsParameter:$axes_maps);
  let hasCustomAssemblyFormat = 1;
  let returnType = "::mlir::tc::AxesMapsRefTy";
  let convertFromStorage = "$_self.getAxesMaps()";
}

def SignsParameter : ArrayRefParameter<"bool", "Signs">;

def SignsAttr : TC_Attr<"Signs", "signs"> {
  let parameters = (ins SignsParameter:$signs);
  let hasCustomAssemblyFormat = 1;
  let returnType = "::mlir::tc::SignsRefTy";
  let convertFromStorage = "$_self.getSigns()";
}

#endif
