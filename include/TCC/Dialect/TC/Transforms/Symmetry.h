#ifndef TCC_DIALECT_TC_TRANSFORMS_SYMMETRY_H
#define TCC_DIALECT_TC_TRANSFORMS_SYMMETRY_H

#include "mlir/IR/Value.h"

#include "mlir/Support/LLVM.h"

#include "TCC/Dialect/Utils/Axis.h"

namespace mlir::tc::symmetry {
// using SymmPairTy = std::pair<int64_t, int64_t>;
using SymmArrayTy = SmallVector<SymmPairTy>;
using ValueSymmetryMapTy = DenseMap<Value, SymmArrayTy>;

void collectSymmInfosOfValue(ValueSymmetryMapTy &symmInfos, Value value);
void cleanEmptyInfoInSymmInfos(ValueSymmetryMapTy &symmInfos);
} // namespace mlir::tc::symmetry

#endif
