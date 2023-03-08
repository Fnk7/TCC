#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "TCC/Dialect/TBLAS/IR/TBLASOps.h"

using namespace mlir;
using namespace mlir::tblas;

#include "TCC/Dialect/TBLAS/IR/TBLASOpsDialect.cpp.inc"

void TBLASDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TCC/Dialect/TBLAS/IR/TBLASOps.cpp.inc"
      >();
}
