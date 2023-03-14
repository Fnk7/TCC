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
using PostionsTy = I64ArrayTy;
using SizeIDs2PosSetMapTy = DenseMap<SizeIDsTy, PostionsTy>;

void insertContractOpPositionIntoGroups(ContractOp ctOp, int64_t pos,
                                        SizeIDs2PosSetMapTy &ctOpGroups) {
  auto axes = getAxes(ctOp);
  auto input = ctOp.getInput();
  auto sizeIDs = getTensorSizeIDs(input);
  auto ctSizeIDs = llvm::to_vector(
      llvm::map_range(axes, [&](auto axis) { return sizeIDs[axis]; }));
  ctOpGroups[ctSizeIDs].emplace_back(pos);
}

Value sinkMulIntoContract(MulOp mulOp) {
  auto rank = getTensorRank(mulOp.getResult());
  auto inputs = mulOp.getInputs();
  int64_t ctPos = -1;
  for (auto [pos, input] : llvm::enumerate(inputs)) {
    if (auto ctOp = input.getDefiningOp<ContractOp>()) {
      assert(ctPos == -1);
      ctPos = pos;
    }
  }
  assert(ctPos != -1);
  auto ctOp = inputs[ctPos].getDefiningOp<ContractOp>();
  auto axes = getAxes(ctOp);
  auto newInputs = llvm::to_vector(inputs);
  auto newAxesMaps = getAxesMaps(mulOp);
  auto i2oAxisMap = getContractOpInput2OutputAxisMap(ctOp);
  applyAxesMapWithEmpty(newAxesMaps[ctPos], i2oAxisMap);
  auto newAxis = rank;
  for (auto &axis : i2oAxisMap) {
    if (axis < 0) {
      axis = newAxis++;
    }
  }
  newInputs[ctPos] = ctOp.getInput();
  newAxesMaps[ctPos] = i2oAxisMap;
  OpBuilder builder(mulOp);
  auto newMulOp = builder.create<MulOp>(mulOp.getLoc(), newInputs, newAxesMaps);
  auto newAxes = llvm::to_vector(llvm::seq<int64_t>(rank, rank + axes.size()));
  auto newCtOp =
      builder.create<ContractOp>(ctOp.getLoc(), newMulOp.getResult(), newAxes);
  return newCtOp.getResult();
}

void mergeAddOfContract(AddOp op);
void mergeAddOfContractAggressive(AddOp op) {
  assert(isElementWiseAddOp(op));
  auto inputs = op.getInputs();
  SizeIDs2PosSetMapTy ctOpGroups;
  for (auto [pos, input] : llvm::enumerate(inputs)) {
    if (auto ctOp = input.getDefiningOp<ContractOp>()) {
      insertContractOpPositionIntoGroups(ctOp, pos, ctOpGroups);
    } else if (auto mulOp = input.getDefiningOp<MulOp>()) {
      SmallVector<ContractOp> ctOps;
      auto inputs = mulOp.getInputs();
      for (auto input : inputs) {
        if (auto ctOp = input.getDefiningOp<ContractOp>()) {
          ctOps.emplace_back(ctOp);
        }
      }
      if (ctOps.size() == 1) {
        auto ctOp = ctOps[0];
        insertContractOpPositionIntoGroups(ctOp, pos, ctOpGroups);
      }
    }
  }
  auto newInputs = llvm::to_vector(inputs);
  for (auto &[sizeIDs, posSet] : ctOpGroups) {
    if (posSet.size() >= 2) {
      for (auto pos : posSet) {
        auto input = inputs[pos];
        if (auto mulOp = input.getDefiningOp<MulOp>()) {
          auto newInput = sinkMulIntoContract(mulOp);
          newInputs[pos] = newInput;
        }
      }
    }
  }
  auto newAxesMaps = getAxesMaps(op);
  OpBuilder builder(op);
  auto newAddOp = builder.create<AddOp>(op.getLoc(), newInputs, newAxesMaps);
  op.replaceAllUsesWith(newAddOp.getResult());
  op.erase();
  mergeAddOfContract(newAddOp);
}

void mergeAddOfContract(AddOp op) {
  assert(isElementWiseAddOp(op));
  auto inputs = op.getInputs();
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
  auto [newValue, newAxesMap] = createAddOp(builder, newInputs, newAxesMaps);
  assert(isOrderedAxesMap(newAxesMap));
  op.replaceAllUsesWith(newValue);
  op.erase();
}
} // namespace

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
      mergeAddOfContractAggressive(addOp);
    }
  }
};

std::unique_ptr<Pass> mlir::tc::createMergeContractPass() {
  return std::make_unique<MergeContractPass>();
}
