#ifndef TCC_DIALECT_TC_UTILS_DEBUG_H
#define TCC_DIALECT_TC_UTILS_DEBUG_H

#include "mlir/IR/Value.h"

namespace mlir::tc::utils {
void dumpValueDefiningChain(Value v);
} // namespace mlir::tc::utils

#endif
