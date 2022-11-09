add_mlir_library(TensorContraction
        ContractionTree.cpp
        Parser.cpp
        MLIRGen.cpp
        LowerToAffineLoops.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include
)

add_subdirectory(TCDialect)
