#include "TCC/Dialect/TC/IR/TCOps.h"
#include "TCC/Dialect/TC/Transforms/Passes.h"
#include "TCC/Dialect/TC/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "llvm/Support/Debug.h"

namespace mlir::tc {
#define GEN_PASS_DEF_GROUPADDWITHSAMEINPUT
#include "TCC/Dialect/TC/Transforms/Passes.h.inc"
} // namespace mlir::tc

#define DEBUG_TYPE "split-contract"

using namespace mlir;
using namespace mlir::tc;

void groupAddOp(AddOp op) {
  auto inputs = op.getInputs();
  DenseMap<Value, SmallVector<int64_t>> valueIdxMap;
  for (auto [idx, input] : llvm::enumerate(inputs)) {
    valueIdxMap[input].emplace_back(idx);
  }
  bool change = false;
  for (auto &e : valueIdxMap) {
    if (e.second.size() >= 2) {
      change = true;
      break;
    }
  }
  if (!change) {
    return;
  }
  auto axesMaps = getAxesMaps(op);
  auto signs = getSigns(op);
  OpBuilder builder(op);
  SmallVector<Value> outerInputs;
  AxesMapsTy outerAxesMaps;
  SignsTy outerSigns;
  for (auto &e : valueIdxMap) {
    auto &indices = e.second;
    if (indices.size() < 2) {
      auto idx = indices[0];
      outerInputs.emplace_back(e.first);
      outerAxesMaps.emplace_back(axesMaps[idx]);
      outerSigns.emplace_back(signs[idx]);
    } else {
      SmallVector<Value> innerInputs(indices.size(), e.first);
      AxesMapsTy innerAxesMaps;
      SignsTy innerSigns;
      for (auto idx : indices) {
        innerAxesMaps.emplace_back(axesMaps[idx]);
        innerSigns.emplace_back(signs[idx]);
      }
      auto addOp = builder.create<AddOp>(op.getLoc(), innerInputs,
                                         innerAxesMaps, innerSigns);
      auto outerAxesMap = getOrderedAxesMap(innerAxesMaps[0].size());
      outerInputs.emplace_back(addOp.getResult());
      outerAxesMaps.emplace_back(outerAxesMap);
      outerSigns.emplace_back(true);
    }
  }
  auto addOp = builder.create<AddOp>(op.getLoc(), outerInputs, outerAxesMaps,
                                     outerSigns);
  op.replaceAllUsesWith(addOp.getResult());
  op.erase();
}

struct GroupAddWithSameInputPass
    : public tc::impl::GroupAddWithSameInputBase<GroupAddWithSameInputPass> {
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
      groupAddOp(addOp);
    }
  }
};

std::unique_ptr<Pass> mlir::tc::createGroupAddWithSameInputPass() {
  return std::make_unique<GroupAddWithSameInputPass>();
}
