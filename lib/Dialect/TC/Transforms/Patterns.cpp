#include "TCC/Dialect/TC/IR/TCOps.h"
#include "TCC/Dialect/TC/IR/TCPatterns.h"
#include "TCC/Dialect/TC/Transforms/Symmetry.h"
#include "TCC/Dialect/TC/Utils/AxesMap.h"
#include "TCC/Dialect/TC/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::tc;
using namespace mlir::tc::utils;

static void dumpValueDefiningChain(Value v, llvm::SetVector<Value> &dumpped) {
  if (dumpped.count(v)) {
    return;
  }
  dumpped.insert(v);
  if (auto arg = v.dyn_cast<BlockArgument>()) {
    auto funcOp = arg.getParentRegion()->getParentOfType<func::FuncOp>();
    auto name = funcOp.getArgAttr(arg.getArgNumber(), "func.name");
    if (name) {
      llvm::errs() << "\%arg" << arg.getArgNumber() << " -> "
                   << name.cast<StringAttr>().getValue() << " ";
      arg.getType().dump();
      llvm::errs() << "\n";
    } else {
      arg.dump();
    }
  } else if (auto *op = v.getDefiningOp()) {
    for (auto v : op->getOperands()) {
      dumpValueDefiningChain(v, dumpped);
    }
    op->dump();
  }
}
static void dumpValueDefiningChain(Value v) {
  llvm::SetVector<Value> dumpped;
  dumpValueDefiningChain(v, dumpped);
}

using InputPosSetTy = llvm::SetVector<int64_t>;
// return true if set1 is subset of set2
static bool isSubsetOf(InputPosSetTy &set1, InputPosSetTy &set2) {
  for (auto pos1 : set1) {
    if (llvm::find(set2, pos1) == set2.end()) {
      return false;
    }
  }
  return true;
}
// return true if set1 intersects with set2
static bool isIntersecting(InputPosSetTy &set1, InputPosSetTy &set2) {
  for (auto pos1 : set1) {
    if (llvm::find(set2, pos1) != set2.end()) {
      return true;
    }
  }
  return false;
}

using MappedInputTy = std::pair<Value, AxesMapTy>;
using MappedInputsTy = SmallVector<MappedInputTy>;
static void mapOriginAxis2NewAxis(AxesMapTy &axesMap) {
  int64_t newAxis = 0;
  for (auto &axis : axesMap) {
    if (axis >= 0) {
      axis = newAxis++;
    }
  }
}
static int64_t getRankFromAxesMaps(AxesMapsTy &axesMaps) {
  int64_t maxAxis = -1;
  for (auto &axesMap : axesMaps) {
    for (auto axis : axesMap) {
      maxAxis = std::max(axis, maxAxis);
    }
  }
  return maxAxis + 1;
}
static AxesMapTy getOrigin2NewAxesMap(AxesMapsTy &axesMaps) {
  auto rank = getRankFromAxesMaps(axesMaps);
  AxesMapTy origin2NewMap(rank, -1);
  for (auto &axesMap : axesMaps) {
    for (auto axis : axesMap) {
      origin2NewMap[axis] = axis;
    }
  }
  mapOriginAxis2NewAxis(origin2NewMap);
  return origin2NewMap;
}
static AxesMapTy getNew2OriginAxesMap(AxesMapTy &origin2NewMap) {
  AxesMapTy new2OriginMap;
  for (auto [originAxis, newAxis] : llvm::enumerate(origin2NewMap)) {
    if (newAxis >= 0) {
      new2OriginMap.emplace_back(originAxis);
    }
  }
  return new2OriginMap;
}
static MappedInputTy createMulOp(OpBuilder &builder, ValueRange inputs,
                                 AxesMapsTy &axesMaps) {
  if (inputs.size() == 1) {
    return MappedInputTy(inputs[0], axesMaps[0]);
  }
  AxesMapTy origin2NewMap = getOrigin2NewAxesMap(axesMaps);
  applyAxesMap(origin2NewMap, axesMaps);
  auto mulOp = builder.create<MulOp>(builder.getUnknownLoc(), inputs, axesMaps);
  auto retAxesMap = getNew2OriginAxesMap(origin2NewMap);
  return MappedInputTy(mulOp.getResult(), retAxesMap);
}
static MappedInputTy createMulOp(OpBuilder &builder,
                                 MappedInputsTy &mappedInputs) {
  if (mappedInputs.size() == 1) {
    return mappedInputs[0];
  }
  SmallVector<Value> inputs;
  AxesMapsTy axesMaps;
  for (auto [input, axesMap] : mappedInputs) {
    inputs.emplace_back(input);
    axesMaps.emplace_back(axesMap);
  }
  return createMulOp(builder, inputs, axesMaps);
}
static MappedInputTy createAddOp(OpBuilder &builder, ValueRange inputs,
                                 AxesMapsTy &axesMaps) {
  if (inputs.size() == 1) {
    return MappedInputTy(inputs[0], axesMaps[0]);
  }
  AxesMapTy origin2NewMap = getOrigin2NewAxesMap(axesMaps);
  SignsTy signs(inputs.size(), true);
  applyAxesMap(origin2NewMap, axesMaps);
  auto mulOp =
      builder.create<AddOp>(builder.getUnknownLoc(), inputs, axesMaps, signs);
  auto retAxesMap = getNew2OriginAxesMap(origin2NewMap);
  return MappedInputTy(mulOp.getResult(), retAxesMap);
}
static MappedInputTy createAddOp(OpBuilder &builder,
                                 MappedInputsTy &mappedInputs) {
  if (mappedInputs.size() == 1) {
    return mappedInputs[0];
  }
  SmallVector<Value> inputs;
  AxesMapsTy axesMaps;
  for (auto [input, axesMap] : mappedInputs) {
    inputs.emplace_back(input);
    axesMaps.emplace_back(axesMap);
  }
  return createAddOp(builder, inputs, axesMaps);
}

struct GroupElementwiseMulPattern : public OpRewritePattern<MulOp> {
  using OpRewritePattern<MulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {
    auto inputs = op.getInputs();
    if (inputs.size() <= 2) {
      return failure();
    }
    auto axesMaps = getAxesMaps(op);
    DenseMap<AxesMapTy, SmallVector<int64_t>> groups;
    for (auto [pos, axesMap] : llvm::enumerate(axesMaps)) {
      auto key = axesMap;
      llvm::sort(key);
      groups[key].emplace_back(pos);
    }
    if (groups.size() == axesMaps.size()) {
      return failure();
    }
    if (groups.size() == 1) {
      return failure();
    }
    SmallVector<Value> newInputs;
    AxesMapsTy newAxesMaps;
    for (auto [_, group] : groups) {
      SmallVector<Value> subInputs;
      AxesMapsTy subAxesMaps;
      for (auto pos : group) {
        subInputs.emplace_back(inputs[pos]);
        subAxesMaps.emplace_back(axesMaps[pos]);
      }
      auto [input, axesMap] = createMulOp(rewriter, subInputs, subAxesMaps);
      newInputs.emplace_back(input);
      newAxesMaps.emplace_back(axesMap);
    }
    rewriter.replaceOpWithNewOp<MulOp>(op, newInputs, newAxesMaps);
    return success();
  }
};

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

namespace factorize {
using FactorTy = std::pair<Value, AxesMapTy>;
using FactorPtr = FactorTy *;
using FactorsHolderTy = SmallVector<std::unique_ptr<FactorTy>>;
using FactorPtrsTy = SmallVector<FactorPtr>;
// collect factorize info from Add of Muls
using AddOfMulFactorsTy = SmallVector<FactorPtrsTy>;
FactorPtr getUniqueFactor(FactorsHolderTy &holder, Value input,
                          AxesMapTy &axesMap) {
  auto key = llvm::find_if(holder, [&](std::unique_ptr<FactorTy> &uniqueKey) {
    return uniqueKey->first == input && uniqueKey->second == axesMap;
  });
  if (key == holder.end()) {
    holder.emplace_back(std::make_unique<FactorTy>(input, axesMap));
    key = &holder.back();
  }
  return key->get();
}
AddOfMulFactorsTy collectAddOfMulFactors(FactorsHolderTy &holder, AddOp op) {
  AddOfMulFactorsTy factors;
  auto inputs = op.getInputs();
  auto axesMaps = getAxesMaps(op);
  for (auto [input, axesMap] : llvm::zip(inputs, axesMaps)) {
    FactorPtrsTy mulFactors;
    if (auto mulOp = input.getDefiningOp<MulOp>()) {
      auto mulInputs = mulOp.getInputs();
      auto mulAxesMaps = getAxesMaps(mulOp);
      for (auto [mulInput, mulAxesMap] : llvm::zip(mulInputs, mulAxesMaps)) {
        applyAxesMap(axesMap, mulAxesMap);
        auto key = getUniqueFactor(holder, mulInput, mulAxesMap);
        mulFactors.emplace_back(key);
      }
    } else {
      auto key = getUniqueFactor(holder, input, axesMap);
      mulFactors.emplace_back(key);
    }
    factors.emplace_back(mulFactors);
  }
  return factors;
}
// Factor to InputPosSet map
using FactorPosSetMapTy = DenseMap<FactorPtr, InputPosSetTy>;
void cleanFactorPosSetMap(FactorPosSetMapTy &map, FactorPtrsTy &toErase) {
  for (auto key : toErase) {
    if (map.count(key)) {
      map.erase(key);
    }
  }
}
FactorPosSetMapTy collectFacorPosSetMap(AddOfMulFactorsTy &factors) {
  FactorPosSetMapTy inputPosSetMap;
  for (auto [pos, mulFactors] : llvm::enumerate(factors)) {
    for (auto factor : mulFactors) {
      inputPosSetMap[factor].insert(pos);
    }
  }
  FactorPtrsTy toErase;
  for (auto &[key, inputPosSet] : inputPosSetMap) {
    if (inputPosSet.size() <= 1) {
      toErase.emplace_back(key);
    }
  }
  cleanFactorPosSetMap(inputPosSetMap, toErase);
  return inputPosSetMap;
}
// Debug Info
// TODO
} // namespace factorize

namespace {
using namespace factorize;
using namespace mlir::tc::symmetry;
struct SymmAddFoldPattern : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    auto result = op.getResult();
    auto inputs = op.getInputs();
    DenseMap<Value, InputPosSetTy> sameInputPos;
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
    ValueSymmInfoMapTy symmInfos;
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
    auto factors = collectAddOfMulFactors(holder, op);
    auto inputPosSetMap = collectFacorPosSetMap(factors);
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

struct MergeAddOfMulPattern : public OpRewritePattern<AddOp> {
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
    auto axesMaps = getAxesMaps(op);
    FactorsHolderTy holder;
    auto factors = collectAddOfMulFactors(holder, op);
    auto inputPosSetMap = collectFacorPosSetMap(factors);
    if (inputPosSetMap.empty()) {
      return failure();
    }
    LLVM_DEBUG(llvm::dbgs() << "<Factorize> Potential factors: "
                            << inputPosSetMap.size() << "\n";);
    // Select one factor in most muls
    {
      FactorPtrsTy toErase;
      for (auto &[key1, inputPosSet1] : inputPosSetMap) {
        for (auto &[key2, inputPosSet2] : inputPosSetMap) {
          if (key1 == key2) {
            continue;
          }
          if (inputPosSet1.size() < inputPosSet2.size()) {
            toErase.emplace_back(key1);
          } else if (inputPosSet1.size() > inputPosSet2.size()) {
            toErase.emplace_back(key2);
          }
        }
      }
      cleanFactorPosSetMap(inputPosSetMap, toErase);
    }
    LLVM_DEBUG(llvm::dbgs() << "<MergeMul> Potential factors: "
                            << inputPosSetMap.size() << "\n";);
    {
      FactorPtrsTy toErase;
      for (auto &[key1, inputPosSet1] : inputPosSetMap) {
        for (auto &[key2, inputPosSet2] : inputPosSetMap) {
          if (key1 == key2) {
            continue;
          }
          if (key1->second.size() < key2->second.size()) {
            toErase.emplace_back(key1);
          } else if (key1->second.size() > key2->second.size()) {
            toErase.emplace_back(key2);
          }
        }
      }
      cleanFactorPosSetMap(inputPosSetMap, toErase);
    }
    LLVM_DEBUG(llvm::dbgs() << "<MergeMul> Potential factors: "
                            << inputPosSetMap.size() << "\n";);
    FactorPtr factor0 = nullptr;
    {
      FactorPtrsTy toErase;
      for (auto &[key1, inputPosSet1] : inputPosSetMap) {
        if (factor0 == nullptr) {
          factor0 = key1;
        } else {
          toErase.emplace_back(key1);
        }
      }
      cleanFactorPosSetMap(inputPosSetMap, toErase);
    }
    LLVM_DEBUG(llvm::dbgs() << "<MergeMul> Potential factors: "
                            << inputPosSetMap.size() << "\n";);
    if (factor0 == nullptr) {
      return failure();
    }
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
    SignsTy signs(newInputs.size(), true);
    rewriter.replaceOpWithNewOp<AddOp>(op, newInputs, newAxesMaps, signs);
    return success();
  }
};
} // namespace

namespace mlir::tc {
void populateGroupElementewiseMulPatterns(RewritePatternSet &patterns,
                                          MLIRContext *context) {
  patterns.add<GroupElementwiseMulPattern>(context);
}
void populateMergeAddOfMulPatterns(RewritePatternSet &patterns,
                                   MLIRContext *context) {
  patterns.add<MergeAddOfMulPattern>(context);
}
void populateSymmAddFoldPatterns(RewritePatternSet &patterns,
                                 MLIRContext *context) {
  patterns.add<SymmAddFoldPattern>(context);
}
} // namespace mlir::tc
