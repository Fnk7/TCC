add_mlir_library(TCCParser ContractionTree.cpp Parser.cpp MLIRGen.cpp
                 ADDITIONAL_HEADER_DIRS ${PROJECT_SOURCE_DIR}/include)
