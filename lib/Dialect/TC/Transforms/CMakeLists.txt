add_mlir_dialect_library(MLIRTCTransforms
  PrintSymmetry.cpp
  MergeContract.cpp
  SplitContract.cpp
  GroupAddWithSameInput.cpp
  AddContractMulTreeAnalysis.cpp
  MergeMul.cpp
  TCOptimizePass.cpp
  TCCSEPass.cpp
  SinkMul.cpp

  DEPENDS
  MLIRTCTransformsIncGen
  )
