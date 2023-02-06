#ifndef TCC_MLIRGEN_H
#define TCC_MLIRGEN_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "TCC/ContractionTree.h"

using ContractionForest = tc::ContractionForest;

namespace mlir::tc {
OwningOpRef<ModuleOp> MLIRGen(MLIRContext &Ctx, ContractionForest &CF);
} // namespace mlir::tc

#endif
