#ifndef TCC_DIALECT_UTILS_AXIS_H
#define TCC_DIALECT_UTILS_AXIS_H

#include "mlir/Support/LLVM.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tc {
using AxisTy = int64_t;
using AxesTy = SmallVector<AxisTy>;
using AxesMapTy = SmallVector<AxisTy>;
using AxesMapsTy = SmallVector<AxesMapTy>;
// ArrayRef
using AxesRefTy = ArrayRef<AxisTy>;
using AxesMapRefTy = ArrayRef<AxisTy>;
using AxesMapsRefTy = ArrayRef<AxesMapTy>;
// AxisPair
using AxisPairTy = std::pair<AxisTy, AxisTy>;
// Symmetry
using SymmPairTy = AxisPairTy;
using SymmArrayTy = SmallVector<SymmPairTy>;
using SymmArrayRefTy = ArrayRef<SymmPairTy>;
using SymmetryMapTy = DenseMap<AxisTy, AxisTy>;
/// TODO: Delete it
using AxesPairsTy = SmallVector<AxesTy>;
} // namespace mlir::tc

namespace llvm {
inline hash_code hash_value(const SmallVectorImpl<mlir::tc::AxisTy> &axes) {
  return hash_value(makeArrayRef(axes));
}
inline hash_code hash_value(const SmallVectorImpl<mlir::tc::AxesMapTy> &maps) {
  return hash_value(makeArrayRef(maps));
}
} // namespace llvm

#endif
