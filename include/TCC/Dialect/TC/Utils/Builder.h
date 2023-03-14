#ifndef TCC_DIALECT_TC_UTILS_BUILDER_H
#define TCC_DIALECT_TC_UTILS_BUILDER_H

#include "TCC/Dialect/TC/IR/Common.h"

#include "mlir/IR/Builders.h"

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
