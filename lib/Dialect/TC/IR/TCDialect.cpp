#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "TCC/Dialect/TC/IR/TCOps.h"

using namespace mlir;
using namespace mlir::tc;

#include "TCC/Dialect/TC/IR/TCOpsDialect.cpp.inc"

void TCDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TCC/Dialect/TC/IR/TCOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "TCC/Dialect/TC/IR/TCAttributes.cpp.inc"
      >();
}

#define GET_ATTRDEF_CLASSES
#include "TCC/Dialect/TC/IR/TCAttributes.cpp.inc"
