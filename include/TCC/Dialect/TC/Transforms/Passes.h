#ifndef TCC_DIALECT_TC_TRANSFORMS_PASSES_H
#define TCC_DIALECT_TC_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir::tc {
std::unique_ptr<Pass> createPrintSymmetryPass();
std::unique_ptr<Pass> createMergeContractPass();
std::unique_ptr<Pass> createSplitContractPass();
std::unique_ptr<Pass> createGroupAddWithSameInputPass();
std::unique_ptr<Pass> createAddContractMulTreeAnalysisPass();
std::unique_ptr<Pass> createMergeMulPass();
std::unique_ptr<Pass> createTCOptimizePass();
std::unique_ptr<Pass> createTCCSEPass();
std::unique_ptr<Pass> createSinkMulPass();
std::unique_ptr<Pass> createLiftMulFromContractPass();
std::unique_ptr<Pass> createSinkMul2Pass();
std::unique_ptr<Pass> createGroupElementwiseMulPass();
std::unique_ptr<Pass> createBeforeCodeGenPass();

void populateGroupElementwiseMulPattern(RewritePatternSet &patterns,
                                         MLIRContext *context);
void populateFactorizePattern(RewritePatternSet &patterns,
                              MLIRContext *context);
void populateSymmAddFoldPattern(RewritePatternSet &patterns,
                                MLIRContext *context);
} // namespace mlir::tc

#endif