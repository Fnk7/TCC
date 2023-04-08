#include "TCC/Dialect/TC/IR/TCOps.h"
#include "TCC/Dialect/TC/Transforms/Passes.h"
#include "TCC/Dialect/TC/Utils/AxesMap.h"
#include "TCC/Dialect/TC/Utils/Builder.h"
#include "TCC/Dialect/TC/Utils/Utils.h"

#include "TCC/Utils/InputPostion.h"
#include "TCC/Utils/SetRelation.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/Support/Debug.h"

namespace mlir::tc {
#define GEN_PASS_DEF_LIFTMUL
#include "TCC/Dialect/TC/Transforms/Passes.h.inc"
} // namespace mlir::tc

#define DEBUG_TYPE "tc-lift-mul"

using namespace mlir;
using namespace mlir::tc;
using namespace mlir::tc::utils;

namespace {
auto applyContractAxesMap(AxesMapRefTy axesMap, AxesMapRefTy innerAxesMap) {
  AxesTy newAxes;
  AxesMapTy outerAxesMap;
  for (auto [iAxis, oAxis] : llvm::enumerate(innerAxesMap)) {
    if (axesMap[oAxis] < 0)
      newAxes.emplace_back(iAxis);
    else
      outerAxesMap.emplace_back(axesMap[oAxis]);
  }
  return std::pair(newAxes, outerAxesMap);
}

bool liftMulFromContract(ContractOp ctOp, OpBuilder &&builder) {
  auto input = ctOp.getInput();
  auto axes = ctOp.getAxes();
  auto mulOp = input.getDefiningOp<MulOp>();
  auto inputs = mulOp.getInputs();
  auto axesMaps = mulOp.getAxesMaps();
  // Collect mul factors to lift
  PositionSetTy toLiftPositions;
  for (auto [position, axesMap] : llvm::enumerate(axesMaps)) {
    if (!isIntersect(axes, axesMap))
      toLiftPositions.insert(position);
  }
  if (toLiftPositions.empty())
    return false;
  auto loc = builder.getUnknownLoc();
  auto rank = getTensorRank(input);
  auto i2oAxisMap = getContractOpInput2OutputAxisMap(axes, rank);
  SmallVector<Value> innerInputs;
  AxesMapsTy innerAxesMaps;
  SmallVector<Value> outerInputs;
  AxesMapsTy outerAxesMaps;
  // Divide into inner and outer
  for (auto [position, input] : llvm::enumerate(inputs)) {
    if (toLiftPositions.count(position)) {
      auto axesMap = axesMaps[position];
      utils::applyAxesMap(i2oAxisMap, axesMap);
      outerInputs.emplace_back(input);
      outerAxesMaps.emplace_back(axesMap);
    } else {
      innerInputs.emplace_back(input);
      innerAxesMaps.emplace_back(axesMaps[position]);
    }
  }
  // Create inner mul
  auto [newInput, innerAxesMap] =
      createMulOp(builder, innerInputs, innerAxesMaps);
  // Get contract and outer mul axes and axes_map
  auto [newAxes, outerAxesMap] = applyContractAxesMap(i2oAxisMap, innerAxesMap);
  // Create contract
  auto newCtOp = builder.create<ContractOp>(loc, newInput, newAxes);
  outerInputs.emplace_back(newCtOp.getResult());
  outerAxesMaps.emplace_back(outerAxesMap);
  // Create outer mul
  auto outerMulOp = builder.create<MulOp>(loc, outerInputs, outerAxesMaps);
  ctOp.replaceAllUsesWith(outerMulOp.getResult());
  return true;
}

struct LiftMulPass : public tc::impl::LiftMulBase<LiftMulPass> {
  void runOnOperation() override {
    bool changed = true;
    while (changed) {
      changed = false;
      SmallVector<ContractOp> worklist;
      for (auto &op : getOperation().front()) {
        if (auto ctOp = dyn_cast_or_null<ContractOp>(&op))
          if (ctOp.getInput().getDefiningOp<MulOp>())
            worklist.emplace_back(ctOp);
      }
      for (auto ctOp : worklist)
        changed = changed || liftMulFromContract(ctOp, OpBuilder(ctOp));
      if (changed) {
        OpPassManager dynamicPM;
        dynamicPM.addPass(createTCCSEPass());
        if (failed(runPipeline(dynamicPM, getOperation())))
          return signalPassFailure();
      }
    }
  }
};
} // namespace

namespace mlir::tc {
std::unique_ptr<Pass> createLiftMulPass() {
  return std::make_unique<LiftMulPass>();
}
} // namespace mlir::tc
