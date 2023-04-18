#ifndef TCC_DIALECT_TBLAS_IR_TBLASOPS_H
#define TCC_DIALECT_TBLAS_IR_TBLASOPS_H

#include "TCC/Dialect/TC/IR/TCOps.h"
#include "TCC/Dialect/Utils/Axis.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "TCC/Dialect/TBLAS/IR/TBLASOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "TCC/Dialect/TBLAS/IR/TBLASOps.h.inc"

#endif
