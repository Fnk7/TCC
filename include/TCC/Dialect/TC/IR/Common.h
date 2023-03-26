#ifndef TCC_DIALECT_TC_IR_COMMON_H
#define TCC_DIALECT_TC_IR_COMMON_H

#include "mlir/Support/LLVM.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tc {
using BoolArrayTy = SmallVector<bool>;
using SignsTy = BoolArrayTy;
using I64ArrayTy = SmallVector<int64_t>;
} // namespace mlir::tc

#endif
