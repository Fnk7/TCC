get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

set(LIBS ${dialect_libs} TensorContraction MLIRContraction)

add_llvm_executable(tcc TCC.cpp)
llvm_update_compile_flags(tcc)
target_link_libraries(tcc PRIVATE ${LIBS})
mlir_check_all_link_libraries(tcc)
