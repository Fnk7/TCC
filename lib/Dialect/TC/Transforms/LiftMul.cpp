#include "TCC/Dialect/TC/IR/TCOps.h"
#include "TCC/Dialect/TC/Transforms/Passes.h"
#include "TCC/Dialect/TC/Utils/AxesMap.h"
#include "TCC/Dialect/TC/Utils/Builder.h"
#include "TCC/Dialect/TC/Utils/Utils.h"

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

struct LiftMulPass : public tc::impl::LiftMulBase<LiftMulPass> {
  void runOnOperation() override {
    bool changed = false;
    do {
      changed = false;
      SmallVector<ContractOp> worklist;
      for (auto &op : getOperation().front()) {
        if (auto ctOp = dyn_cast_or_null<ContractOp>(&op)) {
          if (ctOp.getInput().getDefiningOp<MulOp>()) {
            worklist.emplace_back(ctOp);
          }
        }
      }
      for (auto ctOp : worklist) {
        auto input = ctOp.getInput();
        auto axes = getAxes(ctOp);
        auto rank = getTensorRank(input);
        auto i2oAxisMap = getContractOpInput2OutputAxisMap(axes, rank);
        auto mulOp = input.getDefiningOp<MulOp>();
        auto inputs = mulOp.getInputs();
        auto axesMaps = getAxesMaps(mulOp);
        llvm::SetVector<int64_t> toLift;
        for (auto [pos, axesMap] : llvm::enumerate(axesMaps)) {
          bool canLift = true;
          for (auto axis : axesMap) {
            if (llvm::find(axes, axis) != axes.end()) {
              canLift = false;
            }
          }
          if (canLift) {
            toLift.insert(pos);
          }
        }
        if (toLift.empty()) {
          continue;
        }
        changed = true;
        SmallVector<Value> innerInputs;
        AxesMapsTy innerAxesMaps;
        for (auto [pos, input] : llvm::enumerate(inputs)) {
          if (toLift.count(pos)) {
            continue;
          }
          innerInputs.emplace_back(input);
          innerAxesMaps.emplace_back(axesMaps[pos]);
        }
        OpBuilder builder(ctOp);
        auto [innerInput, innerAxesMap] =
            createMulOp(builder, innerInputs, innerAxesMaps);
        AxesTy newAxes;
        AxesMapTy ctAxesMap;
        for (auto [axis, iAxis] : llvm::enumerate(innerAxesMap)) {
          if (i2oAxisMap[iAxis] < 0) {
            newAxes.emplace_back(axis);
          } else {
            ctAxesMap.emplace_back(i2oAxisMap[iAxis]);
          }
        }
        auto newCtOp =
            builder.create<ContractOp>(ctOp.getLoc(), innerInput, newAxes);
        SmallVector<Value> outerInputs;
        AxesMapsTy outerAxesMaps;
        outerInputs.emplace_back(newCtOp.getResult());
        outerAxesMaps.emplace_back(ctAxesMap);
        for (auto pos : toLift) {
          auto input = inputs[pos];
          auto axesMap = axesMaps[pos];
          for (auto &axis : axesMap) {
            axis = i2oAxisMap[axis];
          }
          outerInputs.emplace_back(input);
          outerAxesMaps.emplace_back(axesMap);
        }
        auto outerMulOp =
            builder.create<MulOp>(mulOp.getLoc(), outerInputs, outerAxesMaps);
        ctOp.replaceAllUsesWith(outerMulOp.getResult());
      }
      if (changed) {
        OpPassManager dynamicPM;
        dynamicPM.addPass(createTCCSEPass());
        if (failed(runPipeline(dynamicPM, getOperation())))
          return signalPassFailure();
      }
    } while (changed);
  }
};

std::unique_ptr<Pass> mlir::tc::createLiftMulPass() {
  return std::make_unique<LiftMulPass>();
}
