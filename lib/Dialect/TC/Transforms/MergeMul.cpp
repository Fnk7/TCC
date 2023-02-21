#include "TCC/Dialect/TC/IR/TCOps.h"
#include "TCC/Dialect/TC/Transforms/Passes.h"
#include "TCC/Dialect/TC/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "llvm/Support/Debug.h"

namespace mlir::tc {
#define GEN_PASS_DEF_MERGEMUL
#include "TCC/Dialect/TC/Transforms/Passes.h.inc"
} // namespace mlir::tc

#define DEBUG_TYPE "merge-mul"

using namespace mlir;
using namespace mlir::tc;

// apply axesMap1 on axesMap2, axesMap2 changed
void applyAxesMap(AxesMapTy &axesMap1, AxesMapTy &axesMap2) {
  for (auto &axis : axesMap2) {
    axis = axesMap1[axis];
  }
}
void applyAxesMap(AxesMapTy &axesMap1, AxesMapsTy &axesMaps) {
  for (auto &axesMap2 : axesMaps) {
    applyAxesMap(axesMap1, axesMap2);
  }
}

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
                   << name.cast<StringAttr>().getValue() << "\n";
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
static int64_t getRankFromAxesMaps(AxesMapsTy &axesMaps) {
  int64_t maxAxis = -1;
  for (auto &axesMap : axesMaps) {
    for (auto axis : axesMap) {
      maxAxis = std::max(axis, maxAxis);
    }
  }
  return maxAxis + 1;
}
static void mapOriginAxis2NewAxis(AxesMapTy &axesMap) {
  int64_t newAxis = 0;
  for (auto &axis : axesMap) {
    if (axis >= 0) {
      axis = newAxis++;
    }
  }
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
AxesMapTy getNew2OriginAxesMap(AxesMapTy &origin2NewMap) {
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
  if (!isElementWiseAddOp(op)) {
    return;
  }
  auto inputs = op.getInputs();
  auto axesMaps = getAxesMaps(op);
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
  if (factorMap2.empty()) {
    return;
  }
  // Not consider intersection
  for (auto key1 : keys) {
    auto &inputSet1 = factorMap2[key1];
    for (auto key2 : keys) {
      auto &inputSet2 = factorMap2[key2];
      if (isIntersecting(inputSet1, inputSet2) &&
          !(isSubsetOf(inputSet1, inputSet2) ||
            isSubsetOf(inputSet2, inputSet1))) {
        llvm::errs() << "Intersection!\n";
        dumpValueDefiningChain(key1->first);
        dumpValueDefiningChain(key2->first);
        return;
      }
    }
  }
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
    llvm::errs() << funcOp.getName() << "\n";
    auto &region = funcOp.getRegion();
    auto &entryBlock = *(region.getBlocks().begin());
    SmallVector<AddOp> workLists;
    for (auto &op : entryBlock) {
      if (auto addOp = dyn_cast_or_null<AddOp>(&op)) {
        workLists.emplace_back(addOp);
      }
    }
    for (auto addOp : workLists) {
      MergeMulOp(addOp);
    }
  }
};

std::unique_ptr<Pass> mlir::tc::createMergeMulPass() {
  return std::make_unique<MergeMulPass>();
}
