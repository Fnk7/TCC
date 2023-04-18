add_mlir_dialect_library(MLIRTBLASTransforms
  DegeneralizePass.cpp

  DEPENDS
  MLIRTBLASTransformsIncGen
  )
