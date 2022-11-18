#include "TCDialect/ContractionOps.h"

using namespace mlir;
using namespace mlir::contraction;

#include "TCDialect/ContractionOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

void ContractionDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TCDialect/ContractionOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Ops
//===----------------------------------------------------------------------===//


#define GET_OP_CLASSES
#include "TCDialect/ContractionOps.cpp.inc"
