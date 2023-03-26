#ifndef TCC_DIALECT_TC_IR_TCOPS_H
#define TCC_DIALECT_TC_IR_TCOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "TCC/Dialect/TC/IR/Common.h"
#include "TCC/Dialect/TC/IR/TCOpsDialect.h.inc"
#include "TCC/Dialect/Utils/Axis.h"
#include "TCC/Dialect/Utils/Tensor.h"

#define GET_OP_CLASSES
#include "TCC/Dialect/TC/IR/TCOps.h.inc"

#define GET_ATTRDEF_CLASSES
#include "TCC/Dialect/TC/IR/TCAttributes.h.inc"

#endif
