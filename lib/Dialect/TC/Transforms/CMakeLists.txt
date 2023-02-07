add_mlir_dialect_library(MLIRTCTransforms
  PrintSymmetry.cpp
  MergeContract.cpp
  SplitContract.cpp

  DEPENDS
  MLIRTCTransformsIncGen
  )
