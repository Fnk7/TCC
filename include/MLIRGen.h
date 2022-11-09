#ifndef TENSORCONTRACTION_MLIRGEN_H
#define TENSORCONTRACTION_MLIRGEN_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "ContractionTree.h"

using ContractionForest = contraction::ContractionForest;

namespace mlir {
namespace contraction {
OwningOpRef<ModuleOp> MLIRGen(MLIRContext &Ctx, ContractionForest &CF);
} // namespace contraction
} // namespace mlir

#endif
