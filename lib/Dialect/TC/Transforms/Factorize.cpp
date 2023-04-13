#include "TCC/Dialect/TC/Transforms/Factorize.h"
#include "TCC/Dialect/TC/IR/TCOps.h"
#include "TCC/Dialect/TC/Utils/AxesMap.h"

namespace mlir::tc {
using namespace utils;
namespace factorize {
// collect factorize info from Add of Muls
FactorPtr getUniqueFactor(FactorsHolderTy &holder, Value input,
                          const AxesMapTy &axesMap) {
  auto key = llvm::find_if(holder, [&](std::unique_ptr<FactorTy> &uniqueKey) {
    return uniqueKey->first == input && uniqueKey->second == axesMap;
  });
  if (key == holder.end()) {
    holder.emplace_back(std::make_unique<FactorTy>(input, axesMap));
    key = &holder.back();
  }
  return key->get();
}
Position2FactorsMap mapAddOpInput2Factors(FactorsHolderTy &holder, AddOp op) {
  Position2FactorsMap pos2FactorsMap;
  auto inputs = op.getInputs();
  auto axesMaps = op.getAxesMaps();
  for (auto [input, axesMap] : llvm::zip(inputs, axesMaps)) {
    FactorPtrsTy mulFactors;
    if (auto mulOp = input.getDefiningOp<MulOp>()) {
      auto mulInputs = mulOp.getInputs();
      auto mulAxesMaps = mulOp.getAxesMapsArray();
      for (auto [mulInput, mulAxesMap] : llvm::zip(mulInputs, mulAxesMaps)) {
        applyAxesMap(axesMap, mulAxesMap);
        auto key = getUniqueFactor(holder, mulInput, mulAxesMap);
        mulFactors.emplace_back(key);
      }
    } else {
      auto key = getUniqueFactor(holder, input, axesMap);
      mulFactors.emplace_back(key);
    }
    pos2FactorsMap.emplace_back(mulFactors);
  }
  return pos2FactorsMap;
}
// Factor to PositionSet (of AddOp) map
Factor2PositionSetMapTy
mapFactor2PositionSet(Position2FactorsMap &pos2FactorsMap) {
  Factor2PositionSetMapTy inputPosSetMap;
  for (auto [pos, mulFactors] : llvm::enumerate(pos2FactorsMap)) {
    for (auto factor : mulFactors) {
      inputPosSetMap[factor].insert(pos);
    }
  }
  FactorPtrsTy toErase;
  for (auto &[key, inputPosSet] : inputPosSetMap)
    if (inputPosSet.size() <= 1)
      toErase.emplace_back(key);
  cleanFactor2PositionSetMap(inputPosSetMap, toErase);
  return inputPosSetMap;
}
void cleanFactor2PositionSetMap(Factor2PositionSetMapTy &map,
                                FactorPtrsTy &toErase) {
  for (auto key : toErase) {
    if (map.count(key)) {
      map.erase(key);
    }
  }
}
} // namespace factorize
} // namespace mlir::tc
