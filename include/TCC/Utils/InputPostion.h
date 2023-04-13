#ifndef TCC_UTILS_INPUTPOSITION_H
#define TCC_UTILS_INPUTPOSITION_H

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tc {
using PositionTy = int64_t;
using PositionsTy = llvm::SmallVector<PositionTy>;
using PositionsRefTy = llvm::ArrayRef<PositionTy>;
using PositionSetTy = llvm::SetVector<PositionTy>;
} // namespace mlir::tc

#endif
