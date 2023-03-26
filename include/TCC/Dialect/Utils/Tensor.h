#ifndef TCC_DIALECT_UTILS_TENSOR_H
#define TCC_DIALECT_UTILS_TENSOR_H

#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir::tc {
using TShapeTy = SmallVector<int64_t>;
using SizeIDsTy = SmallVector<int64_t>;
} // namespace mlir::tc

#endif
