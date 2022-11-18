add_mlir_dialect_library(
        MLIRContraction

        ContractionOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/TCDialect

        DEPENDS
        MLIRContractionOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
        MLIRInferTypeOpInterface
	)
