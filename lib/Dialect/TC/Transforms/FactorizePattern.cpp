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

struct FactorizePattern : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;

  void initialize() {
    /// Signal that this pattern safely handles recursive application.
    setHasBoundedRewriteRecursion();
  }

  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    auto inputs = op.getInputs();
    if (inputs.size() == 2) {
      if (inputs[0] == inputs[1]) {
        return failure();
      }
    }
    auto axesMaps = op.getAxesMapsArray();
    FactorsHolderTy holder;
    auto factors = collectAddOfMulFactors(holder, op);
    auto inputPosSetMap = collectFactorPosSetMap(factors);
    if (inputPosSetMap.empty())
      return failure();
    LLVM_DEBUG(llvm::dbgs() << "<Factorize> Potential factors: "
                            << inputPosSetMap.size() << "\n";);
    // Erase factors like A in A + A + C * D -> 2 * A + C * D
    {
      FactorPtrsTy toErase;
      for (auto &[factor, positions] : inputPosSetMap)
        for (auto position : positions)
          if (factors[position].size() == 1) {
            toErase.emplace_back(factor);
            break;
          }
      cleanFactorPosSetMap(inputPosSetMap, toErase);
    }
    DenseMap<FactorPtr, double> factorizeBenifit;
    DenseMap<MulOp, MulOrders> mulOp2OrdersMap;
    auto [lbs, ubs] = getSizesRanges(op->getParentOfType<func::FuncOp>());
    double lastCost = 1;
    {
      auto sizeIDs = getTensorSizeIDs(inputs[0]);
      for (auto sizeID : sizeIDs)
        lastCost *= ubs[sizeID];
    }
    {
      for (auto &[factor, positions] : inputPosSetMap) {
        double benifit = -lastCost;
        for (auto addPosition : positions) {
          auto input = inputs[addPosition];
          auto mulOp = input.getDefiningOp<MulOp>();
          if (!mulOp2OrdersMap.count(mulOp)) {
            auto sizeIDs = getTensorSizeIDs(mulOp.getResult());
            auto axesMaps = mulOp.getAxesMaps();
            SmallVector<int64_t> sizes;
            for (auto sizeID : sizeIDs)
              sizes.emplace_back(ubs[sizeID]);
            mulOp2OrdersMap[mulOp] = MulOrders(sizes, axesMaps);
          }
          auto &mulOrders = mulOp2OrdersMap[mulOp];
          PositionTy factorPosition;
          for (auto [position, factor2] : llvm::enumerate(factors[addPosition]))
            if (factor == factor2) {
              factorPosition = position;
              break;
            }
          auto originCost = mulOrders.getCost();
          auto restCost = mulOrders.getRestCost(factorPosition);
          benifit += originCost - restCost;
        }
        factorizeBenifit[factor] = benifit;
      }
    }
    LLVM_DEBUG({
      llvm::dbgs() << "AddOp:\n";
      op.dump();
      llvm::dbgs() << "Inputs:\n";
      for (auto input : inputs)
        input.dump();
      llvm::dbgs() << "Benifits:\n";
      for (auto &[factor, positions] : inputPosSetMap) {
        llvm::dbgs().indent(2);
        factor->first.dump();
        llvm::dbgs().indent(4)
            << "Benifit is " << factorizeBenifit[factor] << "\n";
      }
    });
    FactorPtr factor0 = nullptr;
    for (auto [factor, benifit] : factorizeBenifit) {
      if (factorizeBenifit[factor] <= 0)
        continue;
      if (factor0 == nullptr ||
          factorizeBenifit[factor] > factorizeBenifit[factor0])
        factor0 = factor;
    }
    if (factor0 == nullptr) {
      LLVM_DEBUG(llvm::dbgs() << "\n";);
      return failure();
    }
    LLVM_DEBUG({
      llvm::dbgs() << "Selected:\n";
      factor0->first.dump();
      llvm::dbgs() << "\n";
    });
    // Do factorize
    SmallVector<Value> newInputs;
    AxesMapsTy newAxesMaps;
    MappedInputsTy subMappedInputs;
    for (auto [input, axesMap, mulFactors] :
         llvm::zip(inputs, axesMaps, factors)) {
      auto key = llvm::find(mulFactors, factor0);
      if (key == mulFactors.end()) {
        newInputs.emplace_back(input);
        newAxesMaps.emplace_back(axesMap);
      } else {
        MappedInputsTy mulMappedInputs;
        for (auto factor : mulFactors) {
          if (factor != factor0) {
            mulMappedInputs.emplace_back(*factor);
          }
        }
        auto mappedInput = createMulOp(rewriter, mulMappedInputs);
        subMappedInputs.emplace_back(mappedInput);
      }
    }
    MappedInputsTy mulMappedInputs;
    mulMappedInputs.emplace_back(*factor0);
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
