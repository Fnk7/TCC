get_property(MLIR_ALL_LIBS GLOBAL PROPERTY MLIR_ALL_LIBS)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
get_property(MLIR_CONVERSION_LIBS GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS ${dialect_libs} ${translation_libs} ${MLIR_ALL_LIBS} ${MLIR_CONVERSION_LIBS} TCCParser TCCPasses MLIRTBLASDialect)

add_llvm_executable(tcc TCC.cpp)
llvm_update_compile_flags(tcc)
target_link_libraries(tcc PRIVATE ${LIBS})
mlir_check_all_link_libraries(tcc)
