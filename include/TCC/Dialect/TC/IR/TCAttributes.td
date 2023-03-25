#ifndef TCC_DIALECT_TC_IR_TCATTRIBUTE_TD
#define TCC_DIALECT_TC_IR_TCATTRIBUTE_TD

include "./TCBase.td"
include "mlir/IR/AttrTypeBase.td"

class TC_Attr<string attrName, string attrMnemonic,
                   list<Trait> traits = []>
    : AttrDef<TC_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

def AxesMapsAttr : TC_Attr<"AxesMaps", "axesmaps"> {
  let parameters = (ins "int":$value);
  let assemblyFormat = "`<` $value `>`";
}

#endif
