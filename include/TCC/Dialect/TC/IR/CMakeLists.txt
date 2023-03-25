add_mlir_dialect(TCOps tc)
add_mlir_doc(TCOps TCDialect TC/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS TCAttributes.td)
mlir_tablegen(TCAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(TCAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRTCAttributesIncGen)
