add_mlir_dialect_library(TensorContraction
        ContractionTree.cpp
        Parser.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include
)
