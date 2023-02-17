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

template <> struct llvm::DenseMapInfo<SizeIDsTy> {
  static inline SizeIDsTy getEmptyKey() { return SizeIDsTy(); }
  static inline SizeIDsTy getTombstoneKey() { return SizeIDsTy(); }
  static unsigned getHashValue(const SizeIDsTy &Val) {
    return llvm::hash_combine_range(Val.begin(), Val.end());
  }
  static bool isEqual(const SizeIDsTy &LHS, const SizeIDsTy &RHS) {
    return LHS == RHS;
  }
};

// apply axesMap1 on axesMap2, axesMap2 changed
void applyAxesMap(AxesMapTy &axesMap1, AxesMapTy &axesMap2) {
  for (auto &axis : axesMap2) {
    axis = axesMap1[axis];
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

void MergeMulOp(AddOp op) {
  if (!isElementWiseAddOp(op)) {
    return;
  }
  auto inputs = op.getInputs();
  auto axesMaps = getAxesMaps(op);
  using MulOpItem = std::pair<int64_t, MulOp>;
  SmallVector<MulOpItem> worklist;
  for (auto [pos, input] : llvm::enumerate(inputs)) {
    if (auto mulOp = input.getDefiningOp<MulOp>()) {
      worklist.emplace_back(MulOpItem(pos, mulOp));
    }
  }
  if (worklist.empty()) {
    return;
  }
  using UsageTy = std::pair<Value, AxesMapTy>;
  using AddMulPosTy = std::pair<int64_t, int64_t>;
  using AddMulPosesTy = SmallVector<AddMulPosTy>;
  using UsageMapTy = DenseMap<UsageTy, AddMulPosesTy>;
  UsageMapTy usageMap;
  for (auto [addPos, mulOp] : worklist) {
    auto &axesMap = axesMaps[addPos];
    auto mulInputs = mulOp.getInputs();
    auto mulAxesMaps = getAxesMaps(mulOp);
    for (auto [mulPos, z] :
         llvm::enumerate(llvm::zip(mulInputs, mulAxesMaps))) {
      auto [mulInput, mulAxesMap] = z;
      applyAxesMap(axesMap, mulAxesMap);
      UsageTy usage(mulInput, mulAxesMap);
      AddMulPosTy pos(addPos, mulPos);
      usageMap[usage].emplace_back(pos);
    }
  }
  for (auto [addPos, mulOp] : worklist) {
    llvm::errs() << addPos << ": ";
    mulOp.dump();
  }
  for (auto &[usage, poses] : usageMap) {
    if (poses.size() <= 1) {
      continue;
    }
    dumpValueDefiningChain(usage.first);
    for (auto pos : poses) {
      llvm::errs() << "(" << pos.first << ", " << pos.second << "), ";
    }
    llvm::errs() << "\n";
    llvm::errs() << "\n";
  }
}

struct MergeMulPass : public tc::impl::MergeMulBase<MergeMulPass> {
  void runOnOperation() override {
    auto funcOp = getOperation();
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
