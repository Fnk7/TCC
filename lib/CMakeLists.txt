add_mlir_library(TensorContraction
        ContractionTree.cpp
        Parser.cpp
        MLIRGen.cpp
        LowerToAffine.cpp
        ReturnToArgumentPass.cpp
        LowerToLLVM.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include
)

add_subdirectory(TCDialect)
