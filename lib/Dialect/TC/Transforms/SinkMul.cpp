#include "TCC/Dialect/TC/IR/TCOps.h"
#include "TCC/Dialect/TC/Transforms/Passes.h"
#include "TCC/Dialect/TC/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "llvm/Support/Debug.h"

namespace mlir::tc {
#define GEN_PASS_DEF_SINKMUL
#include "TCC/Dialect/TC/Transforms/Passes.h.inc"
} // namespace mlir::tc

#define DEBUG_TYPE "tc-sink-mul"

using namespace mlir;
using namespace mlir::tc;

struct SinkMulPass : public tc::impl::SinkMulBase<SinkMulPass> {
  void runOnOperation() override {
    SmallVector<MulOp> worklist;
    for (auto &op : getOperation().front()) {
      if (auto mulOp = dyn_cast_or_null<MulOp>(&op)) {
        worklist.emplace_back(mulOp);
      }
    }
    while (!worklist.empty()) {
      auto mulOp = worklist.pop_back_val();
      auto inputs = mulOp.getInputs();
      auto axesMaps = getAxesMaps(mulOp);
      SmallVector<int64_t> ctOps;
      SmallVector<llvm::SetVector<int64_t>> sinkInto(inputs.size());
      for (auto [pos, input] : llvm::enumerate(inputs)) {
        if (auto ctOp = input.getDefiningOp<ContractOp>()) {
          ctOps.emplace_back(pos);
        }
      }
      for (auto [pos, axesMap] : llvm::enumerate(axesMaps)) {
        for (auto pos2 : ctOps) {
          if ((int64_t)pos == pos2) {
            continue;
          }
          auto &axesMap2 = axesMaps[pos2];
          bool canSink = true;
          for (auto axis : axesMap) {
            if (llvm::find(axesMap2, axis) == axesMap2.end()) {
              canSink = false;
              break;
            }
          }
          if (canSink) {
            sinkInto[pos2].insert(pos);
          }
        }
      }
      // Simplify sinkInto
      for (auto pos1 : ctOps) {
        auto &set1 = sinkInto[pos1];
        if (set1.empty()) {
          continue;
        }
        for (auto pos2 : ctOps) {
          auto &set2 = sinkInto[pos2];
          if (pos1 == pos2 || set2.empty()) {
            continue;
          }
          if (set2.count(pos1)) {
            set1.clear();
          } else if (set1.count(pos2)) {
            set2.clear();
          } else {
            SmallVector<int64_t> toErase;
            for (auto pos3 : set1) {
              if (set2.count(pos3)) {
                toErase.emplace_back(pos3);
              }
            }
            for (auto pos3 : toErase) {
              set1.erase(llvm::find(set1, pos3));
              set2.erase(llvm::find(set2, pos3));
            }
          }
        }
      }
      bool change = false;
      for (auto ctPos : ctOps) {
        if (!sinkInto[ctPos].empty()) {
          change = true;
          break;
        }
      }
      if (change) {
        SmallVector<Value> mulInputs;
        AxesMapsTy mulAxesMaps;
        for (auto [pos, input] : llvm::enumerate(inputs)) {
          bool push = sinkInto[pos].empty();
          for (auto ctPos : ctOps) {
            if (sinkInto[ctPos].count(pos)) {
              push = false;
              break;
            }
          }
          if (push) {
            mulInputs.emplace_back(input);
            mulAxesMaps.emplace_back(axesMaps[pos]);
          }
        }
        for (auto ctPos : ctOps) {
          if (sinkInto[ctPos].empty()) {
            continue;
          }
          auto ctOp = inputs[ctPos].getDefiningOp<ContractOp>();
          auto &ctAxesMap = axesMaps[ctPos];
          auto input = ctOp.getInput();
          auto axes = getAxes(ctOp);
          auto rank = getTensorRank(input);
          auto i2oAxisMap = getContractOpInput2OutputAxisMap(axes, rank);
          AxesMapTy o2iAxisMap(i2oAxisMap.size() - axes.size());
          for (auto [iAxis, oAxis] : llvm::enumerate(i2oAxisMap)) {
            if (oAxis >= 0) {
              o2iAxisMap[oAxis] = iAxis;
            }
          }
          auto rank2 = getTensorRank(mulOp.getResult());
          AxesMapTy sinkMap(rank2);
          for (auto [axisTo, axisFrom] : llvm::enumerate(ctAxesMap)) {
            sinkMap[axisFrom] = axisTo;
          }
          SmallVector<Value> subInputs;
          AxesMapsTy subAxesMaps;
          subInputs.emplace_back(input);
          subAxesMaps.emplace_back(getOrderedAxesMap(rank));
          for (auto subPos : sinkInto[ctPos]) {
            auto input = inputs[subPos];
            auto axesMap = axesMaps[subPos];
            for (auto &axis : axesMap) {
              axis = sinkMap[axis];
              axis = o2iAxisMap[axis];
            }
            subInputs.emplace_back(input);
            subAxesMaps.emplace_back(axesMap);
          }
          OpBuilder builder(mulOp);
          auto subMulOp =
              builder.create<MulOp>(mulOp.getLoc(), subInputs, subAxesMaps);
          mulInputs.emplace_back(subMulOp.getResult());
          mulAxesMaps.emplace_back(ctAxesMap);
          worklist.emplace_back(subMulOp);
        }
        OpBuilder builder(mulOp);
        auto newMulOp =
            builder.create<MulOp>(mulOp.getLoc(), mulInputs, mulAxesMaps);
        mulOp.replaceAllUsesWith(newMulOp.getResult());
      }
    }
  }
};

std::unique_ptr<Pass> mlir::tc::createSinkMulPass() {
  return std::make_unique<SinkMulPass>();
}
