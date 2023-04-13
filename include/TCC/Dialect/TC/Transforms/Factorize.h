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
Factor2PositionSetMapTy
mapFactor2PositionSet(Position2FactorsMap &pos2FactorsMap);
void cleanFactor2PositionSetMap(Factor2PositionSetMapTy &map,
                                FactorPtrsTy &toErase);
} // namespace factorize
} // namespace mlir::tc

#endif
