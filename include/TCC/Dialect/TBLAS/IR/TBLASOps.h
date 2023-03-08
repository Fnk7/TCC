#ifndef TCC_DIALECT_TBLAS_IR_TBLASOPS_H
#define TCC_DIALECT_TBLAS_IR_TBLASOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::tblas {
using I64ArrayTy = SmallVector<int64_t>;
using AxesTy = I64ArrayTy;
} // namespace mlir::tblas

#include "TCC/Dialect/TBLAS/IR/TBLASOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "TCC/Dialect/TBLAS/IR/TBLASOps.h.inc"

#endif
