#ifndef TCC_DIALECT_TC_TRANSFORMS_FACTORIZE_H
#define TCC_DIALECT_TC_TRANSFORMS_FACTORIZE_H

#include "TCC/Dialect/TC/IR/TCOps.h"
#include "TCC/Dialect/Utils/Axis.h"
#include "TCC/Utils/InputPostion.h"

#include "mlir/IR/Value.h"

namespace mlir::tc {
namespace factorize {
using FactorTy = std::pair<Value, AxesMapTy>;
using FactorPtr = FactorTy *;
using FactorsHolderTy = SmallVector<std::unique_ptr<FactorTy>>;
using FactorPtrsTy = SmallVector<FactorPtr>;
// Map AddOp input position to factors
using Position2FactorsMap = SmallVector<FactorPtrsTy>;
FactorPtr getUniqueFactor(FactorsHolderTy &holder, Value input,
                          const AxesMapTy &axesMap);
Position2FactorsMap mapAddOpInput2Factors(FactorsHolderTy &holder, AddOp op);
// Map factors to input position set
using Factor2PositionSetMapTy = DenseMap<FactorPtr, PositionSetTy>;
inline Factor2PositionSetMapTy
mapFactor2PositionSet(Position2FactorsMap &pos2FactorsMap) {
  Factor2PositionSetMapTy factor2PosSetMap;
  for (auto [pos, mulFactors] : llvm::enumerate(pos2FactorsMap))
    for (auto factor : mulFactors)
      factor2PosSetMap[factor].insert(pos);
  return factor2PosSetMap;
}
inline void
cleanFactor2PositionSetMap(Factor2PositionSetMapTy &factor2PosSetMap,
                           FactorPtrsTy &toErase) {
  for (auto key : toErase)
    if (factor2PosSetMap.count(key))
      factor2PosSetMap.erase(key);
}
inline void
cleanFactor2PositionSetMap(Factor2PositionSetMapTy &factor2PosSetMap) {
  FactorPtrsTy toErase;
  for (auto &[key, inputPosSet] : factor2PosSetMap)
    if (inputPosSet.size() <= 1)
      toErase.emplace_back(key);
  cleanFactor2PositionSetMap(factor2PosSetMap, toErase);
}
} // namespace factorize
} // namespace mlir::tc

#endif
