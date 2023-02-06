#ifndef TCC_DIALECT_TC_IR_TCOPS_H
#define TCC_DIALECT_TC_IR_TCOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

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

#include "TCC/Dialect/TC/IR/TCOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "TCC/Dialect/TC/IR/TCOps.h.inc"

#endif
