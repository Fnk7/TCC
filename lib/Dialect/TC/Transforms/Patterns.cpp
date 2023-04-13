#include "TCC/Dialect/TC/IR/TCOps.h"
#include "TCC/Dialect/TC/IR/TCPatterns.h"
#include "TCC/Dialect/TC/Transforms/Factorize.h"
#include "TCC/Dialect/TC/Transforms/MulOrder.h"
#include "TCC/Dialect/TC/Transforms/Symmetry.h"
#include "TCC/Dialect/TC/Utils/AxesMap.h"
#include "TCC/Dialect/TC/Utils/Builder.h"
#include "TCC/Dialect/TC/Utils/Utils.h"
#include "TCC/Dialect/Utils/FuncUtils.h"
#include "TCC/Utils/InputPostion.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::tc;
using namespace mlir::tc::utils;

#define DEBUG_TYPE "tc-merge-addofmul"

namespace builder {
Value createConstantMulOp(OpBuilder &builder, Value input, double cst) {
  auto loc = input.getLoc();
  auto cstValue = builder.create<ConstantOp>(loc, 0.5).getResult();
  SmallVector<Value> mulInputs;
  AxesMapsTy mulAxesMaps;
  if (auto mulOp = input.getDefiningOp<MulOp>()) {
    auto inputs = mulOp.getInputs();
    mulInputs.append(inputs.begin(), inputs.end());
    mulAxesMaps = getAxesMaps(mulOp);
    mulInputs.emplace_back(cstValue);
    mulAxesMaps.emplace_back(getOrderedAxesMap(0));
    loc = mulOp.getLoc();
  } else {
    mulInputs.emplace_back(cstValue);
    mulInputs.emplace_back(input);
    auto rank = getTensorRank(input);
    mulAxesMaps.emplace_back(getOrderedAxesMap(0));
    mulAxesMaps.emplace_back(getOrderedAxesMap(rank));
  }
  auto newMulOp = builder.create<MulOp>(loc, mulInputs, mulAxesMaps);
  return newMulOp.getResult();
}
} // namespace builder

namespace {
using namespace mlir::tc::factorize;
using namespace mlir::tc::symmetry;
struct SymmAddFoldPattern : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    auto result = op.getResult();
    auto inputs = op.getInputs();
    DenseMap<Value, PositionSetTy> sameInputPos;
    for (auto [pos, input] : llvm::enumerate(inputs)) {
      sameInputPos[input].insert(pos);
    }
    if (sameInputPos.size() == inputs.size()) {
      return failure();
    }
    for (auto &[input, posSet] : sameInputPos) {
      if (posSet.size() >= 3) {
        LLVM_DEBUG(op.print(llvm::dbgs());
                   llvm::dbgs() << "AddOp contains more than 3 same input.\n";);
        return failure();
      }
    }
    ValueSymmetryMapTy symmInfos;
    collectSymmInfosOfValue(symmInfos, result);
    cleanEmptyInfoInSymmInfos(symmInfos);
    if (!symmInfos.count(result)) {
      return failure();
    }
    auto &symmInfo = symmInfos[result];
    if (symmInfo.size() != 1) {
      LLVM_DEBUG(op.print(llvm::dbgs());
                 llvm::dbgs()
                 << "AddOp contains more than 1 symmetry axes pair.\n";);
      return failure();
    }
    auto axesMaps = getAxesMaps(op);
    auto &symmAxisPair = symmInfo[0];
    size_t symmNum = 0;
    SmallVector<int64_t> symmPoses(inputs.size(), -1);
    for (auto &[input, posSet] : sameInputPos) {
      if (posSet.size() == 2) {
        symmNum++;
        auto pos1 = posSet[0];
        auto pos2 = posSet[1];
        auto axesMap1 = axesMaps[pos1];
        auto axesMap2 = axesMaps[pos2];
        std::swap(axesMap1[symmAxisPair.first], axesMap1[symmAxisPair.second]);
        assert(axesMap1 == axesMap2);
        symmPoses[pos1] = pos2;
        symmPoses[pos2] = pos1;
      }
    }
    FactorsHolderTy holder;
    auto factors = mapAddOpInput2Factors(holder, op);
    auto inputPosSetMap = mapFactor2PositionSet(factors);
    SetVector<int64_t> posGroup;
    for (auto &[input, posSet] : inputPosSetMap) {
      if (posSet.size() == inputs.size() - symmNum) {
        posGroup.clear();
        for (auto pos : posSet) {
          if (symmPoses[pos] >= 0) {
            posGroup.insert(pos);
          }
        }
        bool validGroup = true;
        for (auto pos : posGroup) {
          if (posGroup.count(symmPoses[pos])) {
            validGroup = false;
          }
        }
        if (validGroup) {
          break;
        }
      }
    }
    if (posGroup.size() != symmNum) {
      return failure();
    }
    SmallVector<Value> addInputs;
    AxesMapsTy addAxesMaps;
    for (auto pos : llvm::seq<int64_t>(0, inputs.size())) {
      auto input = inputs[pos];
      auto &axesMap = axesMaps[pos];
      if (symmPoses[pos] < 0) {
        // 0.5 * input
        auto newInput = builder::createConstantMulOp(rewriter, input, 0.5);
        addInputs.emplace_back(newInput);
        addAxesMaps.emplace_back(axesMap);
      } else if (posGroup.count(pos)) {
        addInputs.emplace_back(input);
        addAxesMaps.emplace_back(axesMap);
      }
    }
    SignsTy signs(addAxesMaps.size(), true);
    auto subAddOp =
        rewriter.create<AddOp>(op.getLoc(), addInputs, addAxesMaps, signs);
    addInputs.clear();
    addAxesMaps.clear();
    addInputs.emplace_back(subAddOp.getResult());
    addInputs.emplace_back(subAddOp.getResult());
    auto rank = getTensorRank(subAddOp.getResult());
    addAxesMaps.emplace_back(getOrderedAxesMap(rank));
    auto orderAxesMap = getOrderedAxesMap(rank);
    std::swap(orderAxesMap[symmAxisPair.first],
              orderAxesMap[symmAxisPair.second]);
    addAxesMaps.emplace_back(orderAxesMap);
    SignsTy signs2(addAxesMaps.size(), true);
    rewriter.replaceOpWithNewOp<AddOp>(op, addInputs, addAxesMaps, signs2);
    return success();
  }
};
} // namespace

namespace mlir::tc {
void populateSymmAddFoldPattern(RewritePatternSet &patterns,
                                MLIRContext *context) {
  patterns.add<SymmAddFoldPattern>(context);
}
} // namespace mlir::tc
