#ifndef TCC_DIALECT_TC_IR_COMMON_H
#define TCC_DIALECT_TC_IR_COMMON_H

#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir::tc {
using BoolArrayTy = SmallVector<bool>;
using SignsTy = BoolArrayTy;
using I64ArrayTy = SmallVector<int64_t>;
using TShapeTy = I64ArrayTy;
using SizeIDsTy = I64ArrayTy;
using AxesTy = I64ArrayTy;
using AxesMapTy = I64ArrayTy;
using AxesMapsTy = SmallVector<AxesMapTy>;
using AxesPairsTy = SmallVector<AxesTy>;
} // namespace mlir::tc

#endif
