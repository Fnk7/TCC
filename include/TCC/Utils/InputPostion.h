#ifndef TCC_UTILS_INPUTPOSITION_H
#define TCC_UTILS_INPUTPOSITION_H

#include "llvm/ADT/SetVector.h"

namespace mlir::tc {
using PositionTy = int64_t;
using PositionSetTy = llvm::SetVector<PositionTy>;
} // namespace mlir::tc

#endif
