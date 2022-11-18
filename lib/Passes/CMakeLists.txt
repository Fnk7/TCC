add_mlir_library(
        TCCPasses

        LowerToAffine.cpp
        LowerToLLVM.cpp
        ReturnToArgumentPass.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include)
