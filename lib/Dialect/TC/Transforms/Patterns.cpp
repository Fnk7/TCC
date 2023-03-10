#include "TCC/Dialect/TC/IR/TCOps.h"
#include "TCC/Dialect/TC/IR/TCPatterns.h"
#include "TCC/Dialect/TC/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::tc;

template <> struct llvm::DenseMapInfo<I64ArrayTy> {
  static inline I64ArrayTy getEmptyKey() { return I64ArrayTy(1, -2); }
  static inline I64ArrayTy getTombstoneKey() { return I64ArrayTy(1, -2); }
  static unsigned getHashValue(const I64ArrayTy &Val) {
    return llvm::hash_combine_range(Val.begin(), Val.end());
  }
  static bool isEqual(const I64ArrayTy &LHS, const I64ArrayTy &RHS) {
    return LHS == RHS;
  }
};

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

// apply axesMap1 on axesMap2, axesMap2 changed
static void applyAxesMap(AxesMapTy &axesMap1, AxesMapTy &axesMap2) {
  for (auto &axis : axesMap2) {
    axis = axesMap1[axis];
  }
}
static void applyAxesMap(AxesMapTy &axesMap1, AxesMapsTy &axesMaps) {
  for (auto &axesMap2 : axesMaps) {
    applyAxesMap(axesMap1, axesMap2);
  }
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

struct MergeAddOfMulPattern : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;

  using FactorTy = std::pair<Value, AxesMapTy>;
  using FactorPtr = FactorTy *;
  using FactorsHolderTy = SmallVector<std::unique_ptr<FactorTy>>;
  using FactorPtrsTy = SmallVector<FactorPtr>;
  using AddOfMulFactorsTy = SmallVector<FactorPtrsTy>;
  using FactorPosSetMapTy = DenseMap<FactorPtr, InputPosSetTy>;

  void initialize() {
    /// Signal that this pattern safely handles recursive application.
    setHasBoundedRewriteRecursion();
  }

  static FactorPtr getUniqueFactor(FactorsHolderTy &holder, Value input,
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

  static void cleanFactorPosSetMap(FactorPosSetMapTy &map,
                                   FactorPtrsTy &toErase) {
    for (auto key : toErase) {
      if (map.count(key)) {
        map.erase(key);
      }
    }
  }

  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    auto inputs = op.getInputs();
    auto axesMaps = getAxesMaps(op);
    auto mulNum = 0;
    FactorsHolderTy holder;
    AddOfMulFactorsTy factors;
    for (auto [input, axesMap] : llvm::zip(inputs, axesMaps)) {
      FactorPtrsTy mulFactors;
      if (auto mulOp = input.getDefiningOp<MulOp>()) {
        mulNum++;
        auto mulInputs = mulOp.getInputs();
        auto mulAxesMaps = getAxesMaps(mulOp);
        for (auto [mulInput, mulAxesMap] : llvm::zip(mulInputs, mulAxesMaps)) {
          applyAxesMap(axesMap, mulAxesMap);
          FactorTy factor(mulInput, mulAxesMap);
          auto key = getUniqueFactor(holder, mulInput, mulAxesMap);
          mulFactors.emplace_back(key);
        }
      }
      factors.emplace_back(mulFactors);
    }
    if (mulNum <= 1) {
      return failure();
    }
    FactorPosSetMapTy inputPosSetMap;
    {
      for (auto [pos, mulFactors] : llvm::enumerate(factors)) {
        for (auto factor : mulFactors) {
          inputPosSetMap[factor].insert(pos);
        }
      }
      FactorPtrsTy toErase;
      for (auto &[key, inputPosSet] : inputPosSetMap) {
        if (inputPosSet.size() <= 1) {
          toErase.emplace_back(key);
        } else if (key->second.size() == 0) {
          toErase.emplace_back(key);
        }
      }
      cleanFactorPosSetMap(inputPosSetMap, toErase);
    }
    if (inputPosSetMap.empty()) {
      return failure();
    }
    LLVM_DEBUG(llvm::dbgs() << "<MergeMul> Potential factors: "
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

namespace mlir::tc {
void populateGroupElementewiseMulPatterns(RewritePatternSet &patterns,
                                          MLIRContext *context) {
  patterns.add<GroupElementwiseMulPattern>(context);
}
void populateMergeAddOfMulPatterns(RewritePatternSet &patterns,
                                   MLIRContext *context) {
  patterns.add<MergeAddOfMulPattern>(context);
}
} // namespace mlir::tc
