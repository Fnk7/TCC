#include "TCC/Dialect/TC/IR/TCOps.h"
#include "TCC/Dialect/TC/Transforms/Passes.h"
#include "TCC/Dialect/TC/Utils/AxesMap.h"
#include "TCC/Dialect/TC/Utils/Utils.h"

#include "TCC/Dialect/TC/Utils/Builder.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "llvm/Support/Debug.h"

namespace mlir::tc {
#define GEN_PASS_DEF_MERGECONTRACT
#include "TCC/Dialect/TC/Transforms/Passes.h.inc"
} // namespace mlir::tc

#define DEBUG_TYPE "merge-contract"

using namespace mlir;
using namespace mlir::tc;
using namespace mlir::tc::utils;

namespace {
void mergeAddOfContract(AddOp op) {
  assert(isElementWiseAddOp(op));
  auto inputs = op.getInputs();
  using PostionsTy = I64ArrayTy;
  using SizeIDs2PosSetMapTy = DenseMap<SizeIDsTy, PostionsTy>;
  SizeIDs2PosSetMapTy ctOpGroups;
  for (auto [pos, input] : llvm::enumerate(inputs)) {
    if (auto ctOp = input.getDefiningOp<ContractOp>()) {
      auto axes = getAxes(ctOp);
      auto input = ctOp.getInput();
      auto sizeIDs = getTensorSizeIDs(input);
      auto ctSizeIDs = llvm::to_vector(
          llvm::map_range(axes, [&](auto axis) { return sizeIDs[axis]; }));
      ctOpGroups[ctSizeIDs].emplace_back(pos);
    }
  }
  bool doMerge = false;
  for (auto &[sizeIDs, posSet] : ctOpGroups) {
    if (posSet.size() >= 2) {
      doMerge = true;
      break;
    }
  }
  if (!doMerge) {
    return;
  }
  OpBuilder builder(op);
  auto axesMaps = getAxesMaps(op);
  auto rank = axesMaps[0].size();
  SmallVector<Value> newInputs;
  AxesMapsTy newAxesMaps;
  for (auto [input, axesMap] : llvm::zip(inputs, axesMaps)) {
    if (input.getDefiningOp<ContractOp>()) {
      continue;
    } else {
      newInputs.emplace_back(input);
      newAxesMaps.emplace_back(axesMap);
    }
  }
  for (auto &[keySizeIDs, posSet] : ctOpGroups) {
    if (posSet.size() == 1) {
      auto pos = posSet[0];
      newInputs.emplace_back(inputs[pos]);
      newAxesMaps.emplace_back(axesMaps[pos]);
    } else {
      assert(posSet.size() >= 2);
      SmallVector<Value> subInputs;
      AxesMapsTy subAxesMaps;
      for (auto pos : posSet) {
        auto contractOp = inputs[pos].getDefiningOp<ContractOp>();
        auto input = contractOp.getInput();
        auto &axesMap = axesMaps[pos];
        auto axes = getAxes(contractOp);
        auto sizeIDs = getTensorSizeIDs(input);
        auto i2oAxisMap = getContractOpInput2OutputAxisMap(
            axes, axesMap.size() + axes.size());
        applyAxesMapWithEmpty(axesMap, i2oAxisMap);
        auto ctSizeIDs = keySizeIDs;
        for (auto axis : axes) {
          auto sizeID = sizeIDs[axis];
          for (auto [idx, keySizeID] : llvm::enumerate(ctSizeIDs)) {
            if (sizeID == keySizeID) {
              i2oAxisMap[axis] = rank + idx;
              keySizeID = -1;
              break;
            }
          }
        }
        subInputs.emplace_back(input);
        subAxesMaps.emplace_back(i2oAxisMap);
      }
      SignsTy signs(subInputs.size(), true);
      auto subAddOp =
          builder.create<AddOp>(op.getLoc(), subInputs, subAxesMaps, signs);
      auto newAxes =
          llvm::to_vector(llvm::seq<int64_t>(rank, rank + keySizeIDs.size()));
      auto newCtOp = builder.create<ContractOp>(op.getLoc(),
                                                subAddOp.getResult(), newAxes);
      newInputs.emplace_back(newCtOp.getResult());
      newAxesMaps.emplace_back(getOrderedAxesMap(rank));
    }
  }
  auto [newValue, _] = createAddOp(builder, newInputs, newAxesMaps);
  op.replaceAllUsesWith(newValue);
  op.erase();
}
} // namespace

void MergeContractOp(AddOp op) {
  assert(isElementWiseAddOp(op));
  auto inputs = op.getInputs();
  auto axesMaps = getAxesMaps(op);
  auto signs = getSigns(op);
  SmallVector<Value> outerInputs;
  AxesMapsTy outerAxesMaps;
  SignsTy outerSigns;
  using ToMerge = SmallVector<int64_t>;
  using SizeIDs2ToMergeMap = DenseMap<SizeIDsTy, ToMerge>;
  SizeIDs2ToMergeMap map;
  for (auto e : llvm::enumerate(inputs)) {
    auto idx = e.index();
    auto input = e.value();
    if (auto contractOp = input.getDefiningOp<ContractOp>()) {
      auto axes = getAxes(contractOp);
      auto sizeIDs = getTensorSizeIDs(contractOp.getInput());
      SizeIDsTy keySizeIDs;
      for (auto axis : axes) {
        keySizeIDs.emplace_back(sizeIDs[axis]);
      }
      std::sort(keySizeIDs.begin(), keySizeIDs.end());
      map[keySizeIDs].emplace_back(e.index());
    } else {
      outerInputs.emplace_back(input);
      outerAxesMaps.emplace_back(axesMaps[idx]);
      outerSigns.emplace_back(signs[idx]);
    }
  }
  OpBuilder builder(op);
  int64_t rank = axesMaps[0].size();
  for (auto &e : map) {
    if (e.second.size() < 2) {
      auto idx = e.second[0];
      outerInputs.emplace_back(inputs[idx]);
      outerAxesMaps.emplace_back(axesMaps[idx]);
      outerSigns.emplace_back(signs[idx]);
    } else {
      SmallVector<Value> innerInputs;
      AxesMapsTy innerAxesMaps;
      SignsTy innerSigns;
      for (auto idx : e.second) {
        auto contractOp = inputs[idx].getDefiningOp<ContractOp>();
        auto input = contractOp.getInput();
        auto &axesMap = axesMaps[idx];
        auto axes = getAxes(contractOp);
        auto sizeIDs = getTensorSizeIDs(input);
        auto keySizeIDs = e.first;
        auto i2oAxisMap =
            getContractOpInput2OutputAxisMap(axes, rank + axes.size());
        for (auto &axis : i2oAxisMap) {
          if (axis >= 0) {
            axis = axesMap[axis];
          }
        }
        for (auto axis : axes) {
          for (auto [idx, key] : llvm::enumerate(keySizeIDs)) {
            if (key == sizeIDs[axis]) {
              i2oAxisMap[axis] = rank + idx;
              key = -1;
              break;
            }
          }
        }
        innerInputs.emplace_back(input);
        innerAxesMaps.emplace_back(i2oAxisMap);
        innerSigns.emplace_back(signs[idx]);
      }
      auto innerAddOp = builder.create<AddOp>(op.getLoc(), innerInputs,
                                              innerAxesMaps, innerSigns);
      auto r = llvm::seq<int64_t>(rank, rank + e.first.size());
      AxesTy newAxes(r.begin(), r.end());

      auto outerInput =
          builder.create<ContractOp>(op.getLoc(), (Value)innerAddOp, newAxes);
      r = llvm::seq<int64_t>(0, rank);
      AxesMapTy outerAxesMap(r.begin(), r.end());
      outerInputs.emplace_back(outerInput);
      outerAxesMaps.emplace_back(outerAxesMap);
      outerSigns.emplace_back(true);
    }
  }
  if (outerInputs.size() >= 2) {
    auto newAddOp = builder.create<AddOp>(op.getLoc(), outerInputs,
                                          outerAxesMaps, outerSigns);
    op.replaceAllUsesWith(newAddOp.getResult());
  } else {
    op.replaceAllUsesWith(outerInputs[0]);
  }
  op.erase();
}

struct MergeContractPass
    : public tc::impl::MergeContractBase<MergeContractPass> {
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
      MergeContractOp(addOp);
    }
  }
};

std::unique_ptr<Pass> mlir::tc::createMergeContractPass() {
  return std::make_unique<MergeContractPass>();
}
