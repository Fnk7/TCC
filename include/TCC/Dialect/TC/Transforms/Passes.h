#ifndef TCC_DIALECT_TC_TRANSFORMS_PASSES_H
#define TCC_DIALECT_TC_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir::tc {
std::unique_ptr<Pass> createPrintSymmetryPass();
std::unique_ptr<Pass> createMergeContractPass();
std::unique_ptr<Pass> createSplitContractPass();
std::unique_ptr<Pass> createGroupAddWithSameInputPass();
}

#endif