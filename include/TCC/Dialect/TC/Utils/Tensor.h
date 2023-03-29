#ifndef TCC_DIALECT_TC_UTILS_TENSOR_H
#define TCC_DIALECT_TC_UTILS_TENSOR_H

#include "llvm/ADT/SmallVector.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"

#include "TCC/Dialect/Utils/Axis.h"

namespace mlir::tc {
using TShapeTy = SmallVector<int64_t>;
using ShapeRefTy = ArrayRef<int64_t>;
using SizeIDsTy = SmallVector<int64_t>;
using SizeIDsRefTy = ArrayRef<int64_t>;

/// Create tensor with sizeID and symmetry encoding
Type getEncodedTensorType(OpBuilder &builder, ShapeRefTy shape,
                          SizeIDsRefTy sizeIDs, SymmArrayRefTy symmetry);
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
/// Tensor informations
// Get rank
inline int64_t getTensorRank(RankedTensorType RTT) { return RTT.getRank(); }
inline int64_t getTensorRank(Value value) {
  auto RTT = value.getType().cast<RankedTensorType>();
  return getTensorRank(RTT);
}
// Get tensor size information from shape and encoding
inline SizeIDsTy getTensorSizeIDs(RankedTensorType RTT) {
  auto dictAttr = RTT.getEncoding().cast<DictionaryAttr>();
  auto sizeIDsAttr = dictAttr.get("size_id").cast<DenseI64ArrayAttr>();
  return llvm::to_vector(sizeIDsAttr.asArrayRef());
}
inline SizeIDsTy getTensorSizeIDs(Value value) {
  auto RTT = value.getType().cast<RankedTensorType>();
  return getTensorSizeIDs(RTT);
}
inline ShapeRefTy getTensorShape(Value value) {
  auto RTT = value.getType().cast<RankedTensorType>();
  return RTT.getShape();
}
inline auto getTensorSizeInfos(Value value) {
  return std::pair(getTensorShape(value), getTensorSizeIDs(value));
}
} // namespace mlir::tc

#endif
