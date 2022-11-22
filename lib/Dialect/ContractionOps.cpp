#include "TCC/Dialect/ContractionOps.h"

using namespace mlir;
using namespace mlir::contraction;

#include "TCC/Dialect/ContractionOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

void ContractionDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TCC/Dialect/ContractionOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Ops
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "TCC/Dialect/ContractionOps.cpp.inc"
