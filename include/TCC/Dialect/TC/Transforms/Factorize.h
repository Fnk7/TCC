#ifndef TCC_DIALECT_TC_TRANSFORMS_FACTORIZE_H
#define TCC_DIALECT_TC_TRANSFORMS_FACTORIZE_H

#include "TCC/Dialect/Utils/Axis.h"
#include "TCC/Utils/InputPostion.h"

namespace mlir::tc {
namespace factorize {
using FactorTy = std::pair<Value, AxesMapTy>;
using FactorPtr = FactorTy *;
using FactorsHolderTy = SmallVector<std::unique_ptr<FactorTy>>;
using FactorPtrsTy = SmallVector<FactorPtr>;
// Make Add -> Mul into 2 level factors
using AddOfMulFactorsTy = SmallVector<FactorPtrsTy>;
FactorPtr getUniqueFactor(FactorsHolderTy &holder, Value input,
                          AxesMapTy &axesMap);
AddOfMulFactorsTy collectAddOfMulFactors(FactorsHolderTy &holder, AddOp op);
// Map factors to input position set
using FactorPosSetMapTy = DenseMap<FactorPtr, PositionSetTy>;
void cleanFactorPosSetMap(FactorPosSetMapTy &map, FactorPtrsTy &toErase);
FactorPosSetMapTy collectFactorPosSetMap(AddOfMulFactorsTy &factors);
} // namespace factorize
} // namespace mlir::tc

#endif
