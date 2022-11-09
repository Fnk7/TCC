set(LIBS TensorContraction)

add_llvm_executable(parser Parser.cpp)
llvm_update_compile_flags(parser)
target_link_libraries(parser PRIVATE ${LIBS})
mlir_check_all_link_libraries(parser)
