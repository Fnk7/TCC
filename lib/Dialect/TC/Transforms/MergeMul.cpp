#include "TCC/Dialect/TC/IR/TCOps.h"
#include "TCC/Dialect/TC/Transforms/Passes.h"
#include "TCC/Dialect/TC/Transforms/Symmetry.h"
#include "TCC/Dialect/TC/Utils/AxesMap.h"
#include "TCC/Dialect/TC/Utils/Builder.h"
#include "TCC/Dialect/TC/Utils/Debug.h"
#include "TCC/Dialect/TC/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"

namespace mlir::tc {
#define GEN_PASS_DEF_MERGEMUL
#include "TCC/Dialect/TC/Transforms/Passes.h.inc"
} // namespace mlir::tc

#define DEBUG_TYPE "merge-mul"

using namespace mlir;
using namespace mlir::tc;
using namespace mlir::tc::utils;

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

using FactorTy = std::pair<Value, AxesMapTy>;
using AddMulPosTy = std::pair<int64_t, int64_t>;
using AddMulPosesTy = SmallVector<AddMulPosTy>;
using FactorsTy = SmallVector<std::unique_ptr<FactorTy>>;
using FactorPtrsTy = SmallVector<FactorTy *>;
using AddMulFactorsTy = SmallVector<FactorPtrsTy>;
using AddMulTreeTy = SmallVector<FactorPtrsTy>;
using FactorMapTy = DenseMap<FactorTy *, AddMulPosesTy>;
using FactorMap2Ty = DenseMap<FactorTy *, InputPosSetTy>;

MappedInputTy createAddMulTree(OpBuilder &builder,
                               AddMulFactorsTy &addMulFactors,
                               AddMulTreeTy &tree, FactorPtrsTy &parentKeys) {
  MappedInputsTy toAdd;
  llvm::SetVector<FactorTy *> done;
  for (auto [pos, posKeys] : llvm::enumerate(tree)) {
    if (posKeys.size() < parentKeys.size()) {
      continue;
    }
    bool isSubNode = true;
    for (auto [key1, key2] : llvm::zip_first(parentKeys, posKeys)) {
      if (key1 != key2) {
        isSubNode = false;
        break;
      }
    }
    if (!isSubNode) {
      continue;
    }
    if (posKeys.size() > parentKeys.size()) {
      auto nextKey = posKeys[parentKeys.size()];
      if (!done.count(nextKey)) {
        done.insert(nextKey);
        MappedInputsTy toMul;
        parentKeys.emplace_back(nextKey);
        toMul.emplace_back(
            createAddMulTree(builder, addMulFactors, tree, parentKeys));
        parentKeys.pop_back();
        toMul.emplace_back(*nextKey);
        toAdd.emplace_back(createMulOp(builder, toMul));
      }
    } else {
      // TODO: consider pos factors == posKeys
      // exmpale A + A
      FactorPtrsTy toMul;
      for (auto factor : addMulFactors[pos]) {
        if (llvm::find(posKeys, factor) != posKeys.end()) {
          continue;
        }
        toMul.emplace_back(factor);
      }
      assert(toMul.size() && "Currently not support AB + AB.");
      SmallVector<Value> mulInputs;
      AxesMapsTy mulAxesMaps;
      for (auto factor : toMul) {
        mulInputs.emplace_back(factor->first);
        mulAxesMaps.emplace_back(factor->second);
      }
      toAdd.emplace_back(createMulOp(builder, mulInputs, mulAxesMaps));
    }
  }
  return createAddOp(builder, toAdd);
}

FactorTy *getUniqueFactor(FactorsTy &factors, FactorTy &factor) {
  auto unique =
      llvm::find_if(factors, [&](std::unique_ptr<FactorTy> &uniqueKey) {
        return *uniqueKey == factor;
      });
  if (unique == factors.end()) {
    factors.emplace_back(std::make_unique<FactorTy>(factor));
    unique = &factors.back();
  }
  return unique->get();
}

void MergeMulOp(AddOp op) {
  llvm::errs() << "MergeMul Start!\n";
  if (!isElementWiseAddOp(op)) {
    return;
  }
  // For symm add
  if (op.getInputs().size() == 2) {
    auto inputs = op.getInputs();
    if (inputs[0] == inputs[1]) {
      return;
    }
  }
  // for symm add end
  // Symm
  symmetry::ValueSymmetryMapTy symmInfos;
  symmetry::collectSymmInfosOfValue(symmInfos, op.getResult());
  symmetry::cleanEmptyInfoInSymmInfos(symmInfos);
  // Symm end
  auto inputs = op.getInputs();
  auto axesMaps = getAxesMaps(op);
  // Symm
  I64ArrayTy symmPoses(inputs.size(), -1);
  for (auto [pos1, input1] : llvm::enumerate(inputs)) {
    for (auto [pos2, input2] : llvm::enumerate(inputs)) {
      if (pos1 == pos2 || input1 != input2) {
        continue;
      }
      auto axesMap1 = axesMaps[pos1];
      auto axesMap2 = axesMaps[pos2];
      auto &symmInfo = symmInfos[op.getResult()];
      for (auto [axis1, axis2] : symmInfo) {
        auto m1 = axesMap1[axis1];
        auto m2 = axesMap1[axis2];
        axesMap1[axis1] = std::min(m1, m2);
        axesMap1[axis2] = std::max(m1, m2);
        m1 = axesMap2[axis1];
        m2 = axesMap2[axis2];
        axesMap2[axis1] = std::min(m1, m2);
        axesMap2[axis2] = std::max(m1, m2);
      }
      // llvm::errs() << "\n";
      if (llvm::equal(axesMap1, axesMap2)) {
        symmPoses[pos1] = pos2;
      }
    }
  }
  int64_t doAddSplit = 0;
  for (auto pos : symmPoses) {
    if (pos >= 0) {
      doAddSplit += 1;
    }
  }
  // llvm::errs() << "\n";
  // Symm end
  FactorsTy factors;
  AddMulFactorsTy addMulFactors;
  int64_t numMul = 0;
  for (auto [pos, input] : llvm::enumerate(inputs)) {
    FactorPtrsTy mulFactors;
    auto &axesMap = axesMaps[pos];
    if (auto mulOp = input.getDefiningOp<MulOp>()) {
      numMul++;
      auto mulInputs = mulOp.getInputs();
      auto mulAxesMaps = getAxesMaps(mulOp);
      for (auto [mulPos, mulInput] : llvm::enumerate(mulInputs)) {
        auto &mulAxesMap = mulAxesMaps[mulPos];
        applyAxesMap(axesMap, mulAxesMap);
        FactorTy factor(mulInput, mulAxesMap);
        auto key = getUniqueFactor(factors, factor);
        mulFactors.emplace_back(key);
      }
    } else {
      FactorTy factor(input, axesMap);
      auto key = getUniqueFactor(factors, factor);
      mulFactors.emplace_back(key);
    }
    addMulFactors.emplace_back(std::move(mulFactors));
  }
  if (numMul <= 1) {
    return;
  }
  FactorMapTy factorMap;
  for (auto [addPos, mulFactors] : llvm::enumerate(addMulFactors)) {
    for (auto [mulPos, factor] : llvm::enumerate(mulFactors)) {
      AddMulPosTy pos(addPos, mulPos);
      factorMap[factor].emplace_back(pos);
    }
  }
  FactorPtrsTy keys;
  FactorMap2Ty factorMap2;
  for (auto &factor : factors) {
    auto key = factor.get();
    auto &poses = factorMap[key];
    if (poses.size() <= 1) {
      continue;
    }
    keys.emplace_back(key);
    auto &set = factorMap2[key];
    dumpValueDefiningChain(factor->first);
    llvm::errs() << "[";
    for (auto axis : factor->second) {
      llvm::errs() << axis << ", ";
    }
    llvm::errs() << "]\n";
    for (auto pos : poses) {
      set.insert(pos.first);
      llvm::errs() << "(" << pos.first << ", " << pos.second << "), ";
    }
    llvm::errs() << "\n";
  }
  // Symm
  if (doAddSplit) {
    FactorTy *splitKey = nullptr;
    for (auto &[key, set] : factorMap2) {
      int64_t isSplit = 0;
      for (auto pos : set) {
        if (symmPoses[pos] >= 0 && !set.count(symmPoses[pos])) {
          isSplit += 1;
        }
      }
      if (isSplit * 2 == doAddSplit) {
        splitKey = key;
        break;
      }
    }
    if (splitKey) {
      OpBuilder builder(op);
      auto const05 = builder.create<ConstantOp>(op.getLoc(), 0.5).getResult();
      SmallVector<Value> addInputs;
      AxesMapsTy addAxesMaps;
      auto &posSet = factorMap2[splitKey];
      for (auto pos : llvm::seq<int64_t>(0, inputs.size())) {
        auto input = inputs[pos];
        auto &axesMap = axesMaps[pos];
        if (symmPoses[pos] < 0) {
          // 0.5 * input
          if (auto mulOp = input.getDefiningOp<MulOp>()) {
            auto inputs = mulOp.getInputs();
            SmallVector<Value> mulInputs(inputs.begin(), inputs.end());
            auto mulAxesMaps = getAxesMaps(mulOp);
            mulInputs.emplace_back(const05);
            mulAxesMaps.emplace_back(getOrderedAxesMap(0));
            auto newMulOp =
                builder.create<MulOp>(op.getLoc(), mulInputs, mulAxesMaps);
            addInputs.emplace_back(newMulOp.getResult());
            addAxesMaps.emplace_back(axesMap);
          } else {
            SmallVector<Value> mulInputs;
            AxesMapsTy mulAxesMaps;
            mulInputs.emplace_back(input);
            mulInputs.emplace_back(const05);
            mulAxesMaps.emplace_back(getOrderedAxesMap(axesMap.size()));
            mulAxesMaps.emplace_back(getOrderedAxesMap(0));
            auto newMulOp =
                builder.create<MulOp>(op.getLoc(), mulInputs, mulAxesMaps);
            addInputs.emplace_back(newMulOp.getResult());
            addAxesMaps.emplace_back(axesMap);
          }
        } else if (posSet.count(pos)) {
          addInputs.emplace_back(input);
          addAxesMaps.emplace_back(axesMap);
        }
      }
      // TODO signs for symm
      SignsTy signs(addAxesMaps.size(), true);
      auto subAddOp =
          builder.create<AddOp>(op.getLoc(), addInputs, addAxesMaps, signs);
      addInputs.clear();
      addAxesMaps.clear();
      addInputs.emplace_back(subAddOp.getResult());
      addInputs.emplace_back(subAddOp.getResult());
      auto rank = getTensorRank(subAddOp.getResult());
      addAxesMaps.emplace_back(getOrderedAxesMap(rank));
      auto orderAxesMap = getOrderedAxesMap(rank);
      for (auto [axis1, axis2] : symmInfos[op.getResult()]) {
        std::swap(orderAxesMap[axis1], orderAxesMap[axis2]);
      }
      addAxesMaps.emplace_back(orderAxesMap);
      SignsTy signs2(addAxesMaps.size(), true);
      auto newAddOp =
          builder.create<AddOp>(op.getLoc(), addInputs, addAxesMaps, signs2);
      op.replaceAllUsesWith(newAddOp.getResult());
      return MergeMulOp(subAddOp);
    }
  }
  // Symm end
  if (factorMap2.empty()) {
    return;
  }
  // Not consider intersection
  llvm::errs() << "Before Erase intersection keys: " << keys.size() << "\n";
  FactorPtrsTy toErase;
  for (auto key1 : keys) {
    auto &inputSet1 = factorMap2[key1];
    for (auto key2 : keys) {
      auto &inputSet2 = factorMap2[key2];
      if (isIntersecting(inputSet1, inputSet2) &&
          !(isSubsetOf(inputSet1, inputSet2) ||
            isSubsetOf(inputSet2, inputSet1))) {
        llvm::errs() << "Intersection!\n";
        // auto &axesMap1 = key1->second;
        // auto &axesMap2 = key1->second;
        // if (isSubsetOf(axesMap1, axesMap2)) {
        //   llvm::errs() << "Erase key:\n";
        //   dumpValueDefiningChain(key1->first);
        //   key1->first.dump();
        // } else if (isSubsetOf(axesMap2, axesMap1)) {
        //   llvm::errs() << "Erase key:\n";
        //   key2->first.dump();
        //   toErase.emplace_back(key2);
        // } else {
        llvm::errs() << "Erase key:\n";
        key1->first.dump();
        llvm::errs() << "Erase key:\n";
        key2->first.dump();
        toErase.emplace_back(key1);
        toErase.emplace_back(key2);
        // }
      }
    }
  }
  for (auto key : toErase) {
    auto iter = llvm::find(keys, key);
    if (iter != keys.end()) {
      keys.erase(iter);
    }
  }
  llvm::errs() << "After  Erase intersection keys: " << keys.size() << "\n";
  llvm::sort(keys, [&](FactorTy *key1, FactorTy *key2) {
    if (isSubsetOf(factorMap2[key1], factorMap2[key2])) {
      return false;
    } else if (isSubsetOf(factorMap2[key2], factorMap2[key1])) {
      return true;
    }
    return key1 > key2;
  });
  SmallVector<FactorPtrsTy> tree(inputs.size());
  for (auto key : keys) {
    for (auto pos : factorMap2[key]) {
      tree[pos].emplace_back(key);
    }
  }
  for (auto [idx1, nodeKeys] : llvm::enumerate(tree)) {
    llvm::errs() << idx1 << ": ";
    for (auto key1 : nodeKeys) {
      for (auto [idx2, key2] : llvm::enumerate(keys)) {
        if (key1 == key2) {
          llvm::errs() << idx2 << ", ";
        }
      }
    }
    llvm::errs() << "\n";
  }
  // Create Add Mul Add Mul tree
  OpBuilder builder(op);
  FactorPtrsTy parentKeys;
  auto [value, axesMap] =
      createAddMulTree(builder, addMulFactors, tree, parentKeys);
  op.replaceAllUsesWith(value);
}

struct MergeMulPass : public tc::impl::MergeMulBase<MergeMulPass> {
  void runOnOperation() override {
    auto funcOp = getOperation();
    // llvm::errs() << funcOp.getName() << "\n";
    // auto &region = funcOp.getRegion();
    // auto &entryBlock = *(region.getBlocks().begin());
    // SmallVector<AddOp> workLists;
    // for (auto &op : entryBlock) {
    //   if (auto addOp = dyn_cast_or_null<AddOp>(&op)) {
    //     workLists.emplace_back(addOp);
    //   }
    // }
    // for (auto addOp : workLists) {
    //   MergeMulOp(addOp);
    // }
    {
      auto *context = &getContext();
      RewritePatternSet patterns(context);
      populateSymmAddFoldPattern(patterns, context);
      FrozenRewritePatternSet frozenPatterns(std::move(patterns));
      if (failed(applyPatternsAndFoldGreedily(funcOp, frozenPatterns)))
        return signalPassFailure();
    }
    {
      auto *context = &getContext();
      RewritePatternSet patterns(context);
      populateFactorizePattern(patterns, context);
      FrozenRewritePatternSet frozenPatterns(std::move(patterns));
      if (failed(applyPatternsAndFoldGreedily(funcOp, frozenPatterns)))
        return signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> mlir::tc::createMergeMulPass() {
  return std::make_unique<MergeMulPass>();
}
