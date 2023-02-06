set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name TC)
add_public_tablegen_target(MLIRTCTransformsIncGen)

add_mlir_doc(Passes TCPasses TC/ -gen-pass-doc)
