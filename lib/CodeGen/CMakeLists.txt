add_mlir_library(
  TCCCodeGen
  EmitCCode.cpp
  EmitTVM.cpp
  EmitTVMRelay.cpp
  EmitOpenCL.cpp

  ADDITIONAL_HEADER_DIRS ${PROJECT_SOURCE_DIR}/include)
