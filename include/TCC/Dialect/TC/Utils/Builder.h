#ifndef TCC_DIALECT_TC_UTILS_BUILDER_H
#define TCC_DIALECT_TC_UTILS_BUILDER_H

#include "mlir/IR/Builders.h"

#include "TCC/Dialect/TC/Utils/Tensor.h"
#include "TCC/Dialect/Utils/Axis.h"

namespace mlir::tc::utils {
using MappedInputTy = std::pair<Value, AxesMapTy>;
using MappedInputsTy = SmallVector<MappedInputTy>;
MappedInputTy createMulOp(OpBuilder &builder, MappedInputsTy &mappedInputs);
MappedInputTy createMulOp(OpBuilder &builder, ValueRange inputs,
                          AxesMapsTy &axesMaps);
MappedInputTy createAddOp(OpBuilder &builder, MappedInputsTy &mappedInputs);
MappedInputTy createAddOp(OpBuilder &builder, ValueRange inputs,
                          AxesMapsTy &axesMaps);
} // namespace mlir::tc::utils

#endif
