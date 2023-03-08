set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name TBLAS)
add_public_tablegen_target(MLIRTBLASTransformsIncGen)

add_mlir_doc(Passes TBLASPasses TBLAS/ -gen-pass-doc)
