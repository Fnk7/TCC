#ifndef TCC_DIALECT_TBLAS_TRANSFORMS_PASSES_H
#define TCC_DIALECT_TBLAS_TRANSFORMS_PASSES_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir::tc {
void populateSyr2kPattern(RewritePatternSet &patterns, MLIRContext *context);
std::unique_ptr<Pass> createTBLASDegeneralizePass();
std::unique_ptr<Pass> createTBLASSpecialGemmPass();
} // namespace mlir::tc

#endif