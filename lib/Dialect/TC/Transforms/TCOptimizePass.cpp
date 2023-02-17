#include "TCC/Dialect/TC/IR/TCOps.h"
#include "TCC/Dialect/TC/Transforms/Passes.h"
#include "TCC/Dialect/TC/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "llvm/Support/Debug.h"

namespace mlir::tc {
#define GEN_PASS_DEF_TCOPTIMIZE
#include "TCC/Dialect/TC/Transforms/Passes.h.inc"
} // namespace mlir::tc

using namespace mlir;
using namespace mlir::tc;

#define DEBUG_TYPE "tc-opt"

namespace {
using PlanCost = double;
using PlanRoots = SmallVector<Value>;
class Plan {
public:
  PlanCost cost;
  PlanRoots root;
};

using ContractAxesPlans = DenseMap<ContractOp, int64_t>;

class TCOptimizeImpl {
  func::FuncOp funcOp;
  func::ReturnOp returnOp;
  Plan basePlan;

public:
  TCOptimizeImpl(func::FuncOp funcOp) {
    this->funcOp = funcOp;
    auto &block = funcOp.front();
    auto returnOp = func::ReturnOp(block.getTerminator());
    for (auto retValue : returnOp.getOperands()) {
      basePlan.root.emplace_back(retValue);
    }
  }

  void run() {
    // create analysis
    for (auto retValue : basePlan.root) {
      if (auto contractOp = retValue.getDefiningOp<ContractOp>()) {

      } else if (auto addOp = retValue.getDefiningOp<AddOp>()) {
        auto inputs = addOp.getInputs();

      }
    }
  }
};
} // namespace

struct TCOptimizePass : public tc::impl::TCOptimizeBase<TCOptimizePass> {
  void runOnOperation() override { TCOptimizeImpl(getOperation()).run(); }
};

std::unique_ptr<Pass> mlir::tc::createTCOptimizePass() {
  return std::make_unique<TCOptimizePass>();
}
