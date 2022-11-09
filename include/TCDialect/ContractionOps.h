#ifndef TENSORCONTRACTION_DIALECT_CONTRACTIONOPS_H
#define TENSORCONTRACTION_DIALECT_CONTRACTIONOPS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/ParallelCombiningOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "TCDialect/ContractionOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "TCDialect/ContractionOps.h.inc"

#endif // STANDALONE_STANDALONEOPS_H