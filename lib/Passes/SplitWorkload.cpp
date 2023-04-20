#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BlockAndValueMapping.h"
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
auto collectWorkloadInputs(SetVector<Value> &workload) {
  SmallVector<Value> inputs;
  for (auto value : workload) {
    if (value.isa<BlockArgument>())
      inputs.emplace_back(value);
    else {
      auto op = value.getDefiningOp();
      for (auto value : op->getOperands())
        if (!workload.count(value))
          inputs.emplace_back(value);
    }
  }
  return inputs;
}

auto createWorkloadFuncOp(func::FuncOp funcOp, int workloadID,
                          SetVector<Value> &workload, ArrayRef<Value> inputs) {
  OpBuilder builder(funcOp);
  auto result = workload[0];
  LLVM_DEBUG(llvm::dbgs() << "Creating workload for:\n";
             result.print(llvm::dbgs()); llvm::dbgs() << "\n";);
  auto funcName =
      std::string(funcOp.getName()) + "_workload_" + std::to_string(workloadID);
  auto numInputSizes = utils::getNumInputSizes(funcOp);
  SmallVector<Type> inputTypes;
  inputTypes.append(numInputSizes, builder.getIndexType());
  for (auto input : inputs)
    inputTypes.emplace_back(input.getType());
  SmallVector<Type> resultTypes(1, result.getType());
  auto funcType = builder.getFunctionType(inputTypes, resultTypes);
  auto workloadFuncOp =
      builder.create<func::FuncOp>(result.getLoc(), funcName, funcType);
  auto attrNames = utils::getTCFuncAttrNames();
  utils::copyFuncAttributes(funcOp, workloadFuncOp, attrNames);
  auto block = workloadFuncOp.addEntryBlock();
  builder.setInsertionPointToEnd(block);
  BlockAndValueMapping mapper;
  for (auto [argIdx, input] : llvm::enumerate(inputs)) {
    auto blockArgument = block->getArgument(argIdx + numInputSizes);
    mapper.map(input, blockArgument);
  }
  for (auto value : llvm::reverse(workload)) {
    auto op = value.getDefiningOp();
    if (!op)
      continue;
    builder.clone(*op, mapper);
  }
  result = mapper.lookup(result);
  builder.create<ReturnOp>(result.getLoc(), result);
  return workloadFuncOp;
}

auto collectFuncWorkloads(func::FuncOp funcOp) {
  DenseMap<Value, SmallVector<Value>> valueUsing;
  DenseMap<Value, SmallVector<Value>> valueUsedBy;
  for (auto &op : funcOp.front()) {
    for (auto input : op.getOperands())
      for (auto output : op.getResults()) {
        valueUsing[output].emplace_back(input);
        valueUsedBy[input].emplace_back(output);
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
    // Block Argument
    if (!op)
      continue;
    // TBLAS Dialect Operations
    if (op->getDialect() == tblasDialect) {
      for (auto input : valueUsing[output])
        worklist.emplace_back(input);
      continue;
    }
    DenseMap<Value, size_t> valueUseCount;
    auto addValueCount = [&](Value input) -> size_t {
      if (!valueUseCount.count(input))
        valueUseCount[input] = 0;
      valueUseCount[input] += 1;
      return valueUseCount[input];
    };
    auto &workload = workloadMap[output];
    std::function<void(Value)> addWorkload = [&](Value output) {
      LLVM_DEBUG(output.print(llvm::dbgs()); llvm::dbgs() << "\n";);
      workload.insert(output);
      for (auto input : valueUsing[output]) {
        auto count = addValueCount(input);
        if (count == valueUsedBy[input].size())
          addWorkload(input);
      }
    };
    LLVM_DEBUG(llvm::dbgs() << "/--------------------------------\\\n");
    addWorkload(output);
    LLVM_DEBUG(llvm::dbgs() << "\\--------------------------------/\n");
    for (auto [input, count] : valueUseCount) {
      if (count != valueUsedBy[input].size())
        worklist.emplace_back(input);
    }
  }
  return workloadMap;
}

void splitFuncIntoWorkloads(func::FuncOp funcOp) {
  auto workloads = collectFuncWorkloads(funcOp);
  auto numInputSizes = utils::getNumInputSizes(funcOp);
  SmallVector<Value> argSizes;
  for (auto argIdx : llvm::seq<int64_t>(0, numInputSizes))
    argSizes.emplace_back(funcOp.front().getArgument(argIdx));
  auto workloadID = 0;
  OpBuilder builder(funcOp);
  SmallVector<std::pair<func::CallOp, Operation *>> replaces;
  for (auto &[result, workload] : workloads) {
    auto inputs = collectWorkloadInputs(workload);
    auto workloadFuncOp =
        createWorkloadFuncOp(funcOp, workloadID++, workload, inputs);
    auto op = result.getDefiningOp();
    builder.setInsertionPoint(op);
    SmallVector<Value> callOperands;
    callOperands.append(argSizes);
    callOperands.append(inputs);
    auto callOp = builder.create<func::CallOp>(op->getLoc(), workloadFuncOp,
                                               callOperands);
    replaces.emplace_back(callOp, op);
  }
  for (auto [callOp, op] : replaces) {
    op->replaceAllUsesWith(callOp);
    op->erase();
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
