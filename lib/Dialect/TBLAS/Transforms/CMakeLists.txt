add_mlir_dialect_library(MLIRTBLASTransforms
  DegeneralizePass.cpp
  Syr2kPattern.cpp

  DEPENDS
  MLIRTBLASTransformsIncGen
  )
