#ifndef TCC_DIALECT_TBLAS_TRANSFORMS_PASSES_H
#define TCC_DIALECT_TBLAS_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir::tc {
std::unique_ptr<Pass> createTBLASTODOPass();
} // namespace mlir::tc

#endif