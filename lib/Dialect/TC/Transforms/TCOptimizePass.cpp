#include "TCC/Dialect/TC/IR/TCOps.h"
#include "TCC/Dialect/TC/Transforms/Passes.h"
#include "TCC/Dialect/TC/Utils/AxesMap.h"
#include "TCC/Dialect/TC/Utils/Utils.h"

#include "TCC/Dialect/TC/Transforms/ContractOrder.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"

namespace mlir::tc {
#define GEN_PASS_DEF_TCOPTIMIZE
#include "TCC/Dialect/TC/Transforms/Passes.h.inc"
} // namespace mlir::tc

using namespace mlir;
using namespace mlir::tc;
using namespace mlir::tc::utils;

#define DEBUG_TYPE "tc-opt"

double getNaiveCost(func::FuncOp funcOp) {
  double cost = 0;
  auto funcAttr = funcOp->getAttrDictionary();
  auto numInputSizesAttr = funcAttr.get("num_input_sizes").cast<IntegerAttr>();
  auto numInputSizes = numInputSizesAttr.getValue().getSExtValue();
  for (auto &op : funcOp.front()) {
    if (auto mulOp = dyn_cast_or_null<MulOp>(&op)) {
      auto [shape, sizeIDs] = getTensorSizeInfos(mulOp.getResult());
      double mulCost = 1;
      for (auto [size, sizeID] : llvm::zip(shape, sizeIDs)) {
        if (sizeID < numInputSizes) {
          mulCost *= 256;
        } else {
          assert(size >= 1);
          mulCost *= size;
        }
      }
      cost += mulCost;
    }
  }
  return cost;
}

namespace {
void splitContractOp(ContractOp ctOp, ContractOrder &axesOrder) {
  OpBuilder builder(ctOp);
  auto loc = ctOp.getLoc();
  auto input = ctOp.getInput();
  auto axesMap = getOrderedAxesMap(getTensorRank(input));
  for (auto axesRef : llvm::reverse(axesOrder)) {
    AxesTy axes(axesRef);
    llvm::sort(axes);
    for (auto &axis : axes) {
      auto axis2 = axis;
      axis = axesMap[axis];
      axesMap[axis2] = -1;
    }
    auto newCtOp = builder.create<ContractOp>(loc, input, axes);
    input = newCtOp.getResult();
    auto axis = 0;
    for (auto &axis2 : axesMap) {
      if (axis2 >= 0)
        axis2 = axis++;
    }
  }
  ctOp.replaceAllUsesWith(input);
}
} // namespace

struct TCOptimizePass : public tc::impl::TCOptimizeBase<TCOptimizePass> {
  void runOnOperation() override {
    ModuleOp modOp = getOperation();
    SymbolTable symTable(modOp);
    func::FuncOp funcOp = symTable.lookup<func::FuncOp>("CT");
    // 1.1 collect all ContractOp
    SmallVector<ContractOp> ctOpWorklist;
    for (auto &op : funcOp.front())
      if (auto contractOp = dyn_cast_or_null<ContractOp>(&op))
        ctOpWorklist.emplace_back(contractOp);
    LLVM_DEBUG(llvm::errs() << "Number of ContractOp: ";
               llvm::errs() << ctOpWorklist.size() << "\n";);
    // 1.2 Create ContractOrders for each ContractOp
    DenseMap<ContractOp, ContractOrders> ctOpAxesOrders;
    int64_t totalOrders = 1;
    for (auto ctOp : ctOpWorklist) {
      if (auto mulOp = ctOp.getInput().getDefiningOp<MulOp>()) {
        auto axes = ctOp.getAxes();
        if (axes.size() > 1) {
          auto axesMaps = mulOp.getAxesMaps();
          ContractOrders orders(axes, axesMaps);
          totalOrders *= orders.getNumberOfOrders();
          ctOpAxesOrders[ctOp] = std::move(orders);
        }
      }
    }
    LLVM_DEBUG(llvm::errs() << "Total number of ContractOp axes orders: ";
               llvm::errs() << totalOrders << "\n";);
    double minCost = 0;
    func::FuncOp resultFunc;
    for (auto globalOrderID : llvm::seq<int64_t>(0, totalOrders)) {
      OpBuilder builder(funcOp);
      BlockAndValueMapping map;
      auto newFuncOp = cast<func::FuncOp>(builder.clone(*funcOp, map));
      for (auto ctOp : ctOpWorklist) {
        if (ctOpAxesOrders.count(ctOp)) {
          auto &orders = ctOpAxesOrders[ctOp];
          auto orderID = globalOrderID % orders.size();
          globalOrderID = globalOrderID / orders.size();
          auto axesOrder = orders.getOrderByID(orderID);
          auto clonedCtOp = map.lookup(ctOp).getDefiningOp<ContractOp>();
          splitContractOp(clonedCtOp, axesOrder);
        }
      }
      newFuncOp.setName(std::string(funcOp.getName()) +
                        std::to_string(globalOrderID));
      OpPassManager dynamicPM;
      dynamicPM.addPass(createTCCSEPass());

      dynamicPM.addPass(createLiftMulFromContractPass());

      dynamicPM.addPass(createSinkMulPass());
      dynamicPM.addPass(createTCCSEPass());
      dynamicPM.addPass(createMergeContractPass());
      dynamicPM.addPass(createTCCSEPass());
      dynamicPM.addPass(createMergeMulPass());
      dynamicPM.addPass(createTCCSEPass());
      dynamicPM.addPass(createLiftMulFromContractPass());
      dynamicPM.addPass(createTCCSEPass());
      dynamicPM.addPass(createMergeMulPass());
      dynamicPM.addPass(createTCCSEPass());

      dynamicPM.addPass(createSinkMulPass());
      dynamicPM.addPass(createTCCSEPass());
      dynamicPM.addPass(createMergeContractPass());
      dynamicPM.addPass(createTCCSEPass());
      dynamicPM.addPass(createMergeMulPass());
      dynamicPM.addPass(createTCCSEPass());
      dynamicPM.addPass(createLiftMulFromContractPass());
      dynamicPM.addPass(createTCCSEPass());
      dynamicPM.addPass(createMergeMulPass());
      dynamicPM.addPass(createTCCSEPass());

      dynamicPM.addPass(createSinkMulPass());
      dynamicPM.addPass(createTCCSEPass());
      dynamicPM.addPass(createMergeContractPass());
      dynamicPM.addPass(createTCCSEPass());
      dynamicPM.addPass(createMergeMulPass());
      dynamicPM.addPass(createTCCSEPass());
      dynamicPM.addPass(createLiftMulFromContractPass());
      dynamicPM.addPass(createTCCSEPass());
      dynamicPM.addPass(createMergeMulPass());
      dynamicPM.addPass(createTCCSEPass());

      dynamicPM.addPass(createSinkMul2Pass());
      dynamicPM.addPass(createGroupElementwiseMulPass());
      dynamicPM.addPass(createBeforeCodeGenPass());
      dynamicPM.addPass(createCSEPass());
      dynamicPM.addPass(createBeforeCodeGenPass());
      dynamicPM.addPass(createCSEPass());

      if (failed(runPipeline(dynamicPM, newFuncOp)))
        return signalPassFailure();

      auto newCost = getNaiveCost(newFuncOp);
      LLVM_DEBUG(llvm::errs() << "New funcOp cost is " << newCost << "\n";);
      if (!resultFunc) {
        resultFunc = newFuncOp;
        minCost = newCost;
      } else if (newCost < minCost) {
        resultFunc.erase();
        resultFunc = newFuncOp;
        minCost = newCost;
      } else {
        newFuncOp.erase();
      }
    }
    resultFunc.setName(funcOp.getName());
    funcOp.erase();
  }
};

std::unique_ptr<Pass> mlir::tc::createTCOptimizePass() {
  return std::make_unique<TCOptimizePass>();
}
