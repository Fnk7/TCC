#ifndef TENSORCONTRACTION_TCDIALECT_CONTRACTIONOPS_H
#define TENSORCONTRACTION_TCDIALECT_CONTRACTIONOPS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/ParallelCombiningOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "TCC/Dialect/ContractionOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "TCC/Dialect/ContractionOps.h.inc"

#endif // STANDALONE_STANDALONEOPS_H