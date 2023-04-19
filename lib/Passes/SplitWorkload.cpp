#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

#include "TCC/Dialect/TBLAS/IR/TBLASOps.h"
#include "TCC/Dialect/TC/IR/TCOps.h"
#include "TCC/Dialect/TC/Utils/Utils.h"
#include "TCC/Dialect/Utils/FuncUtils.h"
#include "TCC/Passes.h"

#define DEBUG_TYPE "tc-split-workload"

using namespace mlir;
using namespace mlir::tc;

namespace {
void splitFuncIntoWorkloads(func::FuncOp funcOp) {
  DenseMap<Value, SetVector<Value>> valueUsing;
  DenseMap<Value, SetVector<Value>> valueUsedBy;
  for (auto &op : funcOp.front()) {
    for (auto input : op.getOperands())
      for (auto output : op.getResults()) {
        valueUsing[output].insert(input);
        valueUsedBy[input].insert(output);
      }
  }
  auto returnOp =
      llvm::cast_if_present<ReturnOp>(funcOp.front().getTerminator());
  assert(returnOp);
  auto context = funcOp.getContext();
  auto tblasDialect = context->getLoadedDialect<TBLASDialect>();
  SmallVector<Value> worklist;
  for (auto output : returnOp.getOperands())
    worklist.emplace_back(output);
  DenseMap<Value, SetVector<Value>> workloadMap;
  while (!worklist.empty()) {
    auto output = worklist.pop_back_val();
    if (workloadMap.count(output))
      continue;
    auto op = output.getDefiningOp();
    if (op->getDialect() == tblasDialect) {
      workloadMap[output].insert(output);
      continue;
    }
    DenseMap<Value, size_t> valueUseCount;
    auto addValueCount = [&](Value input) -> size_t {
      if (!valueUseCount.count(input))
        valueUseCount[input] = 0;
      valueUseCount[input] += 1;
      return valueUseCount[input];
    };
    std::function<void(Value)> addWorkload = [&](Value output) {
      workloadMap[output].insert(output);
      for (auto input : valueUsing[output]) {
        auto count = addValueCount(input);
        if (count == valueUsedBy[input].size())
          addWorkload(input);
      }
    };
    addWorkload(output);
    for (auto [input, count] : valueUseCount) {
      if (count != valueUsedBy[input].size())
        worklist.emplace_back(input);
    }
  }
  for (auto &[_, workload] : workloadMap) {
    for (auto value : workload)
      LLVM_DEBUG(value.print(llvm::dbgs()); llvm::dbgs() << "\n";);
    LLVM_DEBUG(llvm::dbgs() << "\n");
  }
}

struct SplitWorkloadPass
    : public mlir::PassWrapper<SplitWorkloadPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SplitWorkloadPass)

  void runOnOperation() override {
    auto moduleOp = getOperation();
    auto funcOps = moduleOp.getOps<func::FuncOp>();
    for (auto funcOp : funcOps) {
      if (utils::getWorkloadType(funcOp) == 0)
        splitFuncIntoWorkloads(funcOp);
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> mlir::tc::createSplitWorkloadPass() {
  return std::make_unique<SplitWorkloadPass>();
}
