get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

set(LIBS ${dialect_libs} TCCParser)

add_llvm_executable(mlirgen MLIRGen.cpp)
llvm_update_compile_flags(mlirgen)
target_link_libraries(mlirgen PRIVATE ${LIBS})
mlir_check_all_link_libraries(mlirgen)
