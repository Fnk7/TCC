#include "TCC/Dialect/TC/IR/TCOps.h"
#include "TCC/Dialect/TC/Transforms/Factorize.h"
#include "TCC/Dialect/TC/Transforms/MulOrder.h"
#include "TCC/Dialect/TC/Transforms/Passes.h"
#include "TCC/Dialect/TC/Utils/Builder.h"
#include "TCC/Dialect/Utils/FuncUtils.h"

#include "mlir/IR/PatternMatch.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tc-factorize"

namespace mlir::tc {
using namespace utils;
using namespace factorize;

static MulOrders getMulOrdersOfMulOp(MulOp mulOp) {
  auto [lbs, ubs] = getSizesRanges(mulOp->getParentOfType<func::FuncOp>());
  auto sizeIDs = getTensorSizeIDs(mulOp.getResult());
  SmallVector<int64_t> sizes;
  for (auto sizeID : sizeIDs)
    sizes.emplace_back(ubs[sizeID]);
  auto axesMaps = mulOp.getAxesMaps();
  return MulOrders(sizes, axesMaps);
}

struct FactorizePattern : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;

  void initialize() {
    /// Signal that this pattern safely handles recursive application.
    setHasBoundedRewriteRecursion();
  }

  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    auto inputs = op.getInputs();
    auto numMuls = 0;
    SmallVector<MulOp> mulOps;
    for (auto input : inputs)
      if (auto mulOp = input.getDefiningOp<MulOp>())
        numMuls++;
    if (numMuls <= 1)
      return failure();
    auto axesMaps = op.getAxesMapsArray();
    FactorsHolderTy holder;
    auto pos2FactorsMap = mapAddOpInput2Factors(holder, op);
    auto factor2PosSetMap = mapFactor2PositionSet(pos2FactorsMap);
    cleanFactor2PositionSetMap(factor2PosSetMap);
    if (factor2PosSetMap.empty())
      return failure();
    LLVM_DEBUG(llvm::dbgs() << "<Factorize> Potential factors: "
                            << factor2PosSetMap.size() << "\n";);
    // Erase factors like A in A + A + C * D -> 2 * A + C * D
    {
      FactorPtrsTy toErase;
      for (auto &[factor, positions] : factor2PosSetMap)
        for (auto position : positions)
          if (pos2FactorsMap[position].size() == 1) {
            toErase.emplace_back(factor);
            break;
          }
      cleanFactor2PositionSetMap(factor2PosSetMap, toErase);
    }
    // Get benifit each possible factor
    auto [lbs, ubs] = getSizesRanges(op->getParentOfType<func::FuncOp>());
    DenseMap<MulOp, MulOrders> mulOp2OrdersMap;
    DenseMap<FactorPtr, double> factor2BenifitMap;
    double lastCost = 1;
    auto sizeIDs = getTensorSizeIDs(op.getResult());
    for (auto sizeID : sizeIDs)
      lastCost *= ubs[sizeID];
    for (auto &[factor, positions] : factor2PosSetMap) {
      double benifit = -lastCost;
      for (auto addPos : positions) {
        auto mulOp = inputs[addPos].getDefiningOp<MulOp>();
        if (!mulOp2OrdersMap.count(mulOp))
          mulOp2OrdersMap[mulOp] = getMulOrdersOfMulOp(mulOp);
        auto &mulOrders = mulOp2OrdersMap[mulOp];
        PositionTy factorPosition;
        for (auto [mulPos, factor2] : llvm::enumerate(pos2FactorsMap[addPos]))
          if (factor == factor2) {
            factorPosition = mulPos;
            break;
          }
        auto originCost = mulOrders.getCost();
        auto restCost = mulOrders.getRestCost(factorPosition);
        benifit += originCost - restCost;
      }
      factor2BenifitMap[factor] = benifit;
    }
    LLVM_DEBUG({
      llvm::dbgs() << "AddOp:\n";
      llvm::dbgs() << op << "\n";
      llvm::dbgs() << "Inputs:\n";
      for (auto input : inputs)
        llvm::dbgs() << input << "\n";
      llvm::dbgs() << "Benifits:\n";
      for (auto &[factor, benifit] : factor2BenifitMap) {
        llvm::dbgs().indent(4) << factor->first << "\n";
        llvm::dbgs().indent(8) << "Benifit is " << benifit << "\n";
      }
    });
    // Select the best one
    FactorPtr factor = nullptr;
    for (auto [candidate, benifit] : factor2BenifitMap) {
      if (factor2BenifitMap[candidate] <= 0)
        continue;
      if (factor == nullptr ||
          factor2BenifitMap[candidate] > factor2BenifitMap[factor])
        factor = candidate;
    }
    if (factor == nullptr) {
      LLVM_DEBUG(llvm::dbgs() << "\n";);
      return failure();
    }
    LLVM_DEBUG({
      llvm::dbgs() << "Selected:\n";
      factor->first.dump();
      llvm::dbgs() << "\n";
    });
    // Do factorize
    SmallVector<Value> newInputs;
    AxesMapsTy newAxesMaps;
    MappedInputsTy subMappedInputs;
    for (auto [input, axesMap, mulFactors] :
         llvm::zip(inputs, axesMaps, pos2FactorsMap)) {
      auto key = llvm::find(mulFactors, factor);
      if (key == mulFactors.end()) {
        // Input doesn't contain factor
        newInputs.emplace_back(input);
        newAxesMaps.emplace_back(axesMap);
      } else {
        // Input contains factor
        MappedInputsTy mulMappedInputs;
        for (auto subfactor : mulFactors)
          if (factor != subfactor)
            mulMappedInputs.emplace_back(*subfactor);
        auto mappedInput = createMulOp(rewriter, mulMappedInputs);
        subMappedInputs.emplace_back(mappedInput);
      }
    }
    MappedInputsTy mulMappedInputs;
    mulMappedInputs.emplace_back(*factor);
    auto addMappedInput = createAddOp(rewriter, subMappedInputs);
    mulMappedInputs.emplace_back(addMappedInput);
    auto [subInput, subAxesMap] = createMulOp(rewriter, mulMappedInputs);
    newInputs.emplace_back(subInput);
    newAxesMaps.emplace_back(subAxesMap);
    auto [newValue, _] = createAddOp(rewriter, newInputs, newAxesMaps);
    rewriter.replaceOp(op, newValue);
    return success();
  }
};
void populateFactorizePattern(RewritePatternSet &patterns,
                              MLIRContext *context) {
  patterns.add<FactorizePattern>(context);
}
} // namespace mlir::tc
