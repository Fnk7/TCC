#include "TCC/Dialect/TBLAS/IR/TBLASOps.h"
#include "TCC/Dialect/TBLAS/Utils/Utils.h"

#include "TCC/Dialect/TC/Utils/Utils.h"

#include "mlir/IR/Builders.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::tc;

// Include TableGen files
#define GET_OP_CLASSES
#include "TCC/Dialect/TBLAS/IR/TBLASOps.cpp.inc"
