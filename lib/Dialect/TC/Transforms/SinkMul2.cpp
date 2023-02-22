#include "TCC/Dialect/TC/IR/TCOps.h"
#include "TCC/Dialect/TC/Transforms/Passes.h"
#include "TCC/Dialect/TC/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/Support/Debug.h"

namespace mlir::tc {
#define GEN_PASS_DEF_SINKMUL2
#include "TCC/Dialect/TC/Transforms/Passes.h.inc"
} // namespace mlir::tc

#define DEBUG_TYPE "tc-sink-mul"

using namespace mlir;
using namespace mlir::tc;

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

static bool isElementwiseMulOp(MulOp op) {
  auto axesMaps = getAxesMaps(op);
  assert(!axesMaps.empty());
  auto firstAxesMap = axesMaps[0];
  for (auto &axesMap : axesMaps) {
    if (axesMap.size() != firstAxesMap.size()) {
      return false;
    }
  }
  return true;
}

// axesMap1 is subset of axesMap2
AxesMapTy getAxesMapToAxesMapMap(AxesMapTy &axesMap1, AxesMapTy &axesMap2) {
  AxesMapTy axesMap;
  for (auto axis1 : axesMap1) {
    for (auto [axis, axis2] : llvm::enumerate(axesMap2)) {
      if (axis1 == axis2) {
        axesMap.emplace_back(axis);
        break;
      }
    }
  }
  return axesMap;
}

int64_t sinkIntoAddOp(Value input, AxesMapTy &sinkAxesMap, AddOp addOp);
int64_t sinkIntoCtOp(Value input, AxesMapTy &sinkAxesMap, ContractOp ctOp);
Value doSinkIntoAddOp(Value input, AxesMapTy &sinkAxesMap, AddOp addOp);
Value doSinkIntoCtOp(Value input, AxesMapTy &sinkAxesMap, ContractOp ctOp);
int64_t sinkIntoMulOp(Value input, AxesMapTy &sinkAxesMap, MulOp mulOp,
                      bool inMul = false) {
  auto rank = getTensorRank(input);
  auto rank2 = getTensorRank(mulOp.getResult());
  if (rank == 0 || rank != (int64_t)sinkAxesMap.size() || rank >= rank2) {
    return 0;
  }
  auto inputs = mulOp.getInputs();
  auto axesMaps = getAxesMaps(mulOp);
  for (auto &mulAxesMap : axesMaps) {
    if (mulAxesMap.size() == sinkAxesMap.size()) {
      if (!inMul) {
        bool isElementwise = true;
        for (auto axis : sinkAxesMap) {
          if (llvm::find(mulAxesMap, axis) == mulAxesMap.end()) {
            isElementwise = false;
            break;
          }
        }
        if (isElementwise) {
          return 1;
        }
      }
    }
  }
  int64_t cost = 0;
  for (auto [mulInput, mulAxesMap] : llvm::zip(inputs, axesMaps)) {
    if (mulAxesMap.size() <= sinkAxesMap.size()) {
      continue;
    }
    auto sinkAxesMap2 = getAxesMapToAxesMapMap(sinkAxesMap, mulAxesMap);
    int64_t sinkCost = 0;
    if (auto addOp = mulInput.getDefiningOp<AddOp>()) {
      sinkCost = sinkIntoAddOp(input, sinkAxesMap2, addOp);
    } else if (auto ctOp = mulInput.getDefiningOp<ContractOp>()) {
      sinkCost = sinkIntoCtOp(input, sinkAxesMap2, ctOp);
    }
    if (sinkCost != 0) {
      if (cost == 0 || cost > sinkCost) {
        cost = sinkCost;
      }
    }
  }
  return cost;
}
Value doSinkIntoMulOp(Value input, AxesMapTy &sinkAxesMap, MulOp mulOp,
                      int64_t mulPos = -1) {
  auto inputs = mulOp.getInputs();
  auto axesMaps = getAxesMaps(mulOp);
  for (auto &mulAxesMap : axesMaps) {
    if (mulAxesMap.size() == sinkAxesMap.size()) {
      if (mulPos == -1) {
        bool isElementwise = true;
        for (auto axis : sinkAxesMap) {
          if (llvm::find(mulAxesMap, axis) == mulAxesMap.end()) {
            isElementwise = false;
            break;
          }
        }
        if (isElementwise) {
          SmallVector<Value> newInputs(inputs.begin(), inputs.end());
          AxesMapsTy newAxesMaps = axesMaps;
          newInputs.emplace_back(input);
          newAxesMaps.emplace_back(sinkAxesMap);
          OpBuilder builder(mulOp);
          auto newMulOp =
              builder.create<MulOp>(mulOp.getLoc(), newInputs, newAxesMaps);
          return newMulOp.getResult();
        }
      }
    }
  }
  int64_t sinkPos;
  int64_t cost = 0;
  for (auto [pos, mulInput] : llvm::enumerate(inputs)) {
    auto &mulAxesMap = axesMaps[pos];
    if (mulAxesMap.size() <= sinkAxesMap.size()) {
      continue;
    }
    auto sinkAxesMap2 = getAxesMapToAxesMapMap(sinkAxesMap, mulAxesMap);
    int64_t sinkCost = 0;
    if (auto addOp = mulInput.getDefiningOp<AddOp>()) {
      sinkCost = sinkIntoAddOp(input, sinkAxesMap2, addOp);
    } else if (auto ctOp = mulInput.getDefiningOp<ContractOp>()) {
      sinkCost = sinkIntoCtOp(input, sinkAxesMap2, ctOp);
    }
    if (sinkCost != 0) {
      if (cost == 0 || cost > sinkCost) {
        sinkPos = pos;
        cost = sinkCost;
      }
    }
  }
  assert(cost >= 0);
  auto sinkAxesMap2 = getAxesMapToAxesMapMap(sinkAxesMap, axesMaps[sinkPos]);
  SmallVector<Value> newInputs(inputs.begin(), inputs.end());
  if (auto addOp = inputs[sinkPos].getDefiningOp<AddOp>()) {
    newInputs[sinkPos] = doSinkIntoAddOp(input, sinkAxesMap2, addOp);
  } else if (auto ctOp = inputs[sinkPos].getDefiningOp<ContractOp>()) {
    newInputs[sinkPos] = doSinkIntoCtOp(input, sinkAxesMap2, ctOp);
  }
  if (mulPos >= 0) {
    assert(sinkPos != mulPos);
    newInputs.erase(&newInputs[mulPos]);
    axesMaps.erase(&axesMaps[mulPos]);
  }
  OpBuilder builder(mulOp);
  auto newMulOp = builder.create<MulOp>(mulOp.getLoc(), newInputs, axesMaps);
  return newMulOp.getResult();
}
Value doSinkIntoAddOp(Value input, AxesMapTy &sinkAxesMap, AddOp addOp) {
  // sinkAxesMap : input axis map to addOp result axis is mul axis
  auto inputs = addOp.getInputs();
  auto axesMaps = getAxesMaps(addOp);
  auto signs = getSigns(addOp);
  SmallVector<Value> newInputs;
  for (auto [addInput, addAxesMap] : llvm::zip(inputs, axesMaps)) {
    auto sinkAxesMap2 = getAxesMapToAxesMapMap(sinkAxesMap, addAxesMap);
    if (auto mulOp = addInput.getDefiningOp<MulOp>()) {
      newInputs.emplace_back(doSinkIntoMulOp(input, sinkAxesMap2, mulOp));
    } else if (auto ctOp = addInput.getDefiningOp<ContractOp>()) {
      newInputs.emplace_back(doSinkIntoCtOp(input, sinkAxesMap2, ctOp));
    }
  }
  assert(inputs.size() == newInputs.size());
  OpBuilder builder(addOp);
  auto newAddOp =
      builder.create<AddOp>(addOp.getLoc(), newInputs, axesMaps, signs);
  return newAddOp.getResult();
}
int64_t sinkIntoAddOp(Value input, AxesMapTy &sinkAxesMap, AddOp addOp) {
  // sinkAxesMap : input axis map to addOp result axis is mul axis
  auto rank = getTensorRank(input);
  auto rank2 = getTensorRank(addOp.getResult());
  if (rank == 0 || rank != (int64_t)sinkAxesMap.size() || rank >= rank2) {
    return 0;
  }
  auto inputs = addOp.getInputs();
  for (auto addInput : inputs) {
    if (!addInput.getDefiningOp<MulOp>() &&
        !addInput.getDefiningOp<ContractOp>()) {
      return 0;
    }
  }
  auto axesMaps = getAxesMaps(addOp);
  int64_t totalCost = 0;
  for (auto [addInput, addAxesMap] : llvm::zip(inputs, axesMaps)) {
    auto sinkAxesMap2 = getAxesMapToAxesMapMap(sinkAxesMap, addAxesMap);
    int64_t sinkCost = 0;
    if (auto mulOp = addInput.getDefiningOp<MulOp>()) {
      sinkCost = sinkIntoMulOp(input, sinkAxesMap2, mulOp);
    } else if (auto ctOp = addInput.getDefiningOp<ContractOp>()) {
      sinkCost = sinkIntoCtOp(input, sinkAxesMap2, ctOp);
    }
    if (sinkCost == 0) {
      return 0;
    }
    totalCost += sinkCost;
  }
  return totalCost;
}
int64_t sinkIntoCtOp(Value input, AxesMapTy &sinkAxesMap, ContractOp ctOp) {
  auto rank = getTensorRank(input);
  auto rank2 = getTensorRank(ctOp.getResult());
  if (rank == 0 || rank != (int64_t)sinkAxesMap.size() || rank >= rank2) {
    return 0;
  }
  auto ctInput = ctOp.getInput();
  auto axes = getAxes(ctOp);
  auto ctAxesMap =
      getContractOpInput2OutputAxisMap(axes, getTensorRank(ctInput));
  if (auto mulOp = ctInput.getDefiningOp<MulOp>()) {
    auto sinkAxesMap2 = getAxesMapToAxesMapMap(sinkAxesMap, ctAxesMap);
    auto sinkCost = sinkIntoMulOp(input, sinkAxesMap2, mulOp);
    return sinkCost;
  }
  return 0;
}
Value doSinkIntoCtOp(Value input, AxesMapTy &sinkAxesMap, ContractOp ctOp) {
  auto ctInput = ctOp.getInput();
  auto axes = getAxes(ctOp);
  auto ctAxesMap =
      getContractOpInput2OutputAxisMap(axes, getTensorRank(ctInput));
  auto mulOp = ctInput.getDefiningOp<MulOp>();
  assert(mulOp);
  auto sinkAxesMap2 = getAxesMapToAxesMapMap(sinkAxesMap, ctAxesMap);
  auto newInput = doSinkIntoMulOp(input, sinkAxesMap2, mulOp);
  OpBuilder builder(ctOp);
  auto newCtOp = builder.create<ContractOp>(ctOp.getLoc(), newInput, axes);
  return newCtOp.getResult();
}

bool sinkIntoSubMuls(MulOp op) {
  auto inputs = op.getInputs();
  auto axesMaps = getAxesMaps(op);
  for (auto [pos, input] : llvm::enumerate(inputs)) {
    auto &axesMap = axesMaps[pos];
    auto cost = sinkIntoMulOp(input, axesMap, op, true);
    if (cost == 0) {
      continue;
    }
    auto result = doSinkIntoMulOp(input, axesMap, op, pos);
    op.replaceAllUsesWith(result);
    return true;
  }
  return false;
}

struct SinkMul2Pass : public tc::impl::SinkMul2Base<SinkMul2Pass> {
  void runOnOperation() override {
    while (true) {
      SmallVector<MulOp> worklist;
      for (auto &op : getOperation().front()) {
        if (auto mulOp = dyn_cast_or_null<MulOp>(&op)) {
          if (!isElementwiseMulOp(mulOp)) {
            worklist.emplace_back(mulOp);
          }
        }
      }
      bool finished = true;
      for (auto mulOp : worklist) {
        if (sinkIntoSubMuls(mulOp)) {
          finished = false;
          break;
        }
      }
      if (finished) {
        break;
      }
      OpPassManager dynamicPM;
      dynamicPM.addPass(createTCCSEPass());
      if (failed(runPipeline(dynamicPM, getOperation())))
        return signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> mlir::tc::createSinkMul2Pass() {
  return std::make_unique<SinkMul2Pass>();
}
