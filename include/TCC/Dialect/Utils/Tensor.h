#ifndef TCC_DIALECT_UTILS_TENSOR_H
#define TCC_DIALECT_UTILS_TENSOR_H

#include "llvm/ADT/SmallVector.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"

namespace mlir::tc {
using TShapeTy = SmallVector<int64_t>;
using ShapeRefTy = ArrayRef<int64_t>;
using SizeIDsTy = SmallVector<int64_t>;
using SizeIDsRefTy = ArrayRef<int64_t>;

/// Create tensor with sizeID and symmetry encoding
inline Type getEncodedTensorType(OpBuilder &builder, ShapeRefTy shape,
                                 SizeIDsRefTy sizeIDs) {
  auto encoding = SmallVector<NamedAttribute>({
      builder.getNamedAttr("size_id", builder.getDenseI64ArrayAttr(sizeIDs)),
  });
  auto encodingAttr = builder.getDictionaryAttr(encoding);
  return RankedTensorType::get(shape, builder.getF64Type(), encodingAttr);
}
// create zero rank tensor
inline Type getZeroRankedTensorType(OpBuilder &builder) {
  TShapeTy shape;
  SizeIDsTy sizeIDs;
  return getEncodedTensorType(builder, shape, sizeIDs);
}
// Get tensor sizeID from
inline SizeIDsTy getTensorSizeIDs(RankedTensorType RTT) {
  auto dictAttr = RTT.getEncoding().cast<DictionaryAttr>();
  auto sizeIDsAttr = dictAttr.get("size_id").cast<DenseI64ArrayAttr>();
  return llvm::to_vector(sizeIDsAttr.asArrayRef());
}
} // namespace mlir::tc

#endif
