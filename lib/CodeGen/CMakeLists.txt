add_mlir_library(
  TCCCodeGen
  EmitCCode.cpp
  EmitTVM.cpp

  ADDITIONAL_HEADER_DIRS ${PROJECT_SOURCE_DIR}/include)
