#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "TCC/Dialect/TC/IR/TCOps.h"

using namespace mlir;
using namespace mlir::tc;

#include "TCC/Dialect/TC/IR/TCOpsDialect.cpp.inc"

void TCDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TCC/Dialect/TC/IR/TCOps.cpp.inc"
      >();
}
