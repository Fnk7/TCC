#ifndef TCC_DIALECT_TC_IR_TCOPS_H
#define TCC_DIALECT_TC_IR_TCOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "TCC/Dialect/TC/IR/TCOpsDialect.h.inc"
#include "TCC/Dialect/Utils/Axis.h"
#include "TCC/Dialect/Utils/Tensor.h"

namespace mlir::tc {
using SignsTy = SmallVector<bool>;
using SignsRefTy = ArrayRef<bool>;
using I64ArrayTy = SmallVector<int64_t>;
} // namespace mlir::tc

#define GET_ATTRDEF_CLASSES
#include "TCC/Dialect/TC/IR/TCAttributes.h.inc"

#define GET_OP_CLASSES
#include "TCC/Dialect/TC/IR/TCOps.h.inc"

#endif
