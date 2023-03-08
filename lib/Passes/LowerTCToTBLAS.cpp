#include "TCC/Dialect/TC/IR/TCOps.h"
#include "TCC/Dialect/TC/IR/TCPatterns.h"
#include "TCC/Dialect/TC/Transforms/Passes.h"
#include "TCC/Dialect/TC/Utils/Utils.h"

#include "TCC/Dialect/TBLAS/IR/TBLASOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
// #include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "convert-tc-2-tblas"

using namespace mlir;

struct ConvertTCToTBLASPass
    : public PassWrapper<ConvertTCToTBLASPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertTCToTBLASPass)
  void runOnOperation() override {
    SmallVector<tc::ContractOp> worklist;
    for (auto &op : getOperation().front()) {
      if (auto ctOp = dyn_cast_or_null<tc::ContractOp>(&op)) {
        worklist.emplace_back(ctOp);
      }
    }
    for (auto ctOp : worklist) {
      auto input = ctOp.getInput();
      if (auto mulOp = input.getDefiningOp<tc::MulOp>()) {
        auto inputs = mulOp.getInputs();
        if (inputs.size() == 2) {
          auto rank = tc::getTensorRank(input);
          auto axes = tc::getAxes(ctOp);
          auto axesMaps = tc::getAxesMaps(mulOp);
          auto &axesMap1 = axesMaps[0];
          auto &axesMap2 = axesMaps[1];
          auto inverseMap1 = tc::inverseAxesMap(axesMap1, rank);
          auto inverseMap2 = tc::inverseAxesMap(axesMap2, rank);
          tblas::AxesTy restAxes1;
          tblas::AxesTy restAxes2;
          tblas::AxesTy batchAxes1;
          tblas::AxesTy batchAxes2;
          tblas::AxesTy contractAxes1;
          tblas::AxesTy contractAxes2;
          for (auto axis : llvm::seq<int64_t>(0, rank)) {
            if (llvm::find(axes, axis) != axes.end()) {
              assert(inverseMap1[axis] >= 0);
              assert(inverseMap2[axis] >= 0);
              contractAxes1.emplace_back(inverseMap1[axis]);
              contractAxes2.emplace_back(inverseMap2[axis]);
            } else {
              if (inverseMap1[axis] >= 0 && inverseMap2[axis] >= 0) {
                batchAxes1.emplace_back(inverseMap1[axis]);
                batchAxes2.emplace_back(inverseMap2[axis]);
              } else if (inverseMap1[axis] >= 0) {
                restAxes1.emplace_back(inverseMap1[axis]);
              } else {
                assert(inverseMap2[axis] >= 0);
                restAxes2.emplace_back(inverseMap2[axis]);
              }
            }
          }
          auto ctI2OAxesMap = tc::getContractOpInput2OutputAxisMap(axes, rank);
          tc::AxesMapTy permuteMap;
          for (auto [iAxis1, iAxis2] : llvm::zip(batchAxes1, batchAxes2)) {
            auto oAxis1 = axesMap1[iAxis1];
            auto oAxis2 = axesMap2[iAxis2];
            assert(oAxis1 == oAxis2);
            auto oAxis = ctI2OAxesMap[oAxis1];
            permuteMap.emplace_back(oAxis);
          }
          for (auto iAxis : restAxes1) {
            auto oAxis = axesMap1[iAxis];
            oAxis = ctI2OAxesMap[oAxis];
            permuteMap.emplace_back(oAxis);
          }
          for (auto iAxis : restAxes2) {
            auto oAxis = axesMap2[iAxis];
            oAxis = ctI2OAxesMap[oAxis];
            permuteMap.emplace_back(oAxis);
          }
          if (restAxes1.size() && restAxes2.size()) {
            OpBuilder builder(ctOp);
            auto bacthMatMulOp = builder.create<tblas::BatchMatMulOp>(
                ctOp.getLoc(), inputs[0], inputs[1], batchAxes1, batchAxes2,
                restAxes1, restAxes2, contractAxes1, contractAxes2);
            auto permuteOp = builder.create<tc::PermuteOp>(
                ctOp.getLoc(), bacthMatMulOp.getResult(), permuteMap);
            ctOp.replaceAllUsesWith(permuteOp.getResult());
          }
        }
      }
    }
  }
};

namespace mlir::tc {
std::unique_ptr<Pass> createConvertTCToTBLASPass() {
  return std::make_unique<ConvertTCToTBLASPass>();
}
} // namespace mlir::tc