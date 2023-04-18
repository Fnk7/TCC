#ifndef TCC_DIALECT_TBLAS_TRANSFORMS_PASSES_H
#define TCC_DIALECT_TBLAS_TRANSFORMS_PASSES_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir::tc {
std::unique_ptr<Pass> createTBLASDegeneralizePass();
void populateSyr2kPattern(RewritePatternSet &patterns, MLIRContext *context);
} // namespace mlir::tc

#endif