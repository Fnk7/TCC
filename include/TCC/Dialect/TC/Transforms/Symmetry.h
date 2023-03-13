#ifndef TCC_DIALECT_TC_TRANSFORMS_SYMMETRY_H
#define TCC_DIALECT_TC_TRANSFORMS_SYMMETRY_H

#include "mlir/IR/Value.h"

#include "mlir/Support/LLVM.h"

namespace mlir::tc::symmetry {
using SymmAxisPairTy = std::pair<int64_t, int64_t>;
using SymmetryInfoTy = SmallVector<SymmAxisPairTy>;
using ValueSymmInfoMapTy = DenseMap<Value, SymmetryInfoTy>;

void collectSymmInfosOfValue(ValueSymmInfoMapTy &symmInfos, Value value);
void cleanEmptyInfoInSymmInfos(ValueSymmInfoMapTy &symmInfos);
} // namespace mlir::tc::symmetry

#endif
