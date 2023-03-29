#ifndef TCC_DIALECT_UTILS_TENSOR_H
#define TCC_DIALECT_UTILS_TENSOR_H

#include "llvm/ADT/SmallVector.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"

namespace mlir::tc {
using TShapeTy = SmallVector<int64_t>;
using SizeIDsTy = SmallVector<int64_t>;

inline Type getEncodedTensorType(OpBuilder &builder, ArrayRef<int64_t> shape,
                                 SizeIDsTy &encoding) {
  auto encodingAttr = builder.getI64ArrayAttr(encoding);
  return RankedTensorType::get(shape, builder.getF64Type(), encodingAttr);
}
inline Type getZeroRankedTensorType(OpBuilder &builder) {
  TShapeTy shape;
  SizeIDsTy sizeIDs;
  return getEncodedTensorType(builder, shape, sizeIDs);
}
} // namespace mlir::tc

#endif
