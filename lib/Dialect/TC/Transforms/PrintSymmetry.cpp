#include "TCC/Dialect/TC/IR/TCOps.h"
#include "TCC/Dialect/TC/Transforms/Passes.h"
#include "TCC/Dialect/TC/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::tc {
#define GEN_PASS_DEF_PRINTSYMMETRY
#include "TCC/Dialect/TC/Transforms/Passes.h.inc"
} // namespace mlir::tc

using namespace mlir;
using namespace mlir::tc;

struct PrintSymmetryPass
    : public tc::impl::PrintSymmetryBase<PrintSymmetryPass> {
  void runOnOperation() override {
    // TODO: split into functions
    using SymmAxisPair = std::pair<int64_t, int64_t>;
    using SymmInfo = SmallVector<SymmAxisPair>;
    using ValueSymmInfoMap = DenseMap<Value, SymmInfo>;
    using Factor = std::pair<Value, int64_t>;
    using Factors = SmallVector<Factor>;
    using AxisFactorsMap = DenseMap<int64_t, Factors>;
    ValueSymmInfoMap symmInfos;
    auto funcOp = getOperation();
    auto &region = funcOp.getRegion();
    auto &entryBlock = *(region.getBlocks().begin());
    for (auto &op : entryBlock) {
      if (auto mulOp = dyn_cast_or_null<MulOp>(&op)) {
        auto inputs = mulOp.getInputs();
        auto axesMaps = getAxesMaps(mulOp);
        // map to factors
        AxisFactorsMap axisFactors;
        for (auto [input, axesMap] : llvm::zip(inputs, axesMaps)) {
          for (auto e : llvm::enumerate(axesMap)) {
            axisFactors[e.value()].emplace_back(Factor(input, e.index()));
          }
        }
        // sort factors
        for (auto &e : axisFactors) {
          auto &factors = e.second;
          std::sort(factors.begin(), factors.end(), [](Factor &f1, Factor &f2) {
            auto t1 = f1.first;
            auto t2 = f2.first;
            if (t1 != t2) {
              return compareTensorOrder(t1, t2);
            } else {
              return f1.second < f2.second;
            }
          });
        }
        // compare and insert
        Value res = mulOp.getResult();
        int64_t dim = axisFactors.size();
        for (int64_t axis1 = 0; axis1 < dim; axis1++) {
          for (int64_t axis2 = axis1 + 1; axis2 < dim; axis2++) {
            if (axisFactors[axis1] == axisFactors[axis2]) {
              symmInfos[res].emplace_back(SymmAxisPair(axis1, axis2));
            }
          }
        }
      } else if (auto contractOp = dyn_cast_or_null<ContractOp>(&op)) {
        auto input = contractOp.getInput();
        if (symmInfos.count(input)) {
          auto dim = getTensorRank(input);
          auto axes = getAxes(contractOp);
          auto i2oAxisMap = getContractOpInput2OutputAxesMap(axes, dim);
          SymmInfo symmInfo;
          for (auto &pair : symmInfos[input]) {
            auto axis1 = pair.first;
            auto axis2 = pair.second;
            if (i2oAxisMap[axis1] >= 0 && i2oAxisMap[axis2] >= 0) {
              axis1 = i2oAxisMap[axis1];
              axis2 = i2oAxisMap[axis2];
              symmInfo.emplace_back(axis1, axis2);
            }
          }
          if (symmInfo.size()) {
            symmInfos[contractOp.getResult()] = symmInfo;
          }
        }
      } else if (auto addOp = dyn_cast_or_null<AddOp>(&op)) {
        auto inputs = addOp.getInputs();
        auto axesMaps = getAxesMaps(addOp);
        // map to factors
        AxisFactorsMap axisFactors;
        for (auto [input, axesMap] : llvm::zip(inputs, axesMaps)) {
          for (auto e : llvm::enumerate(axesMap)) {
            int64_t axis = e.index();
            if (symmInfos.count(input)) {
              for (auto &pair : symmInfos[input]) {
                if (pair.second == axis) {
                  axis = pair.first;
                }
              }
            }
            axisFactors[e.value()].emplace_back(Factor(input, axis));
          }
        }
        // sort factors
        for (auto &e : axisFactors) {
          auto &factors = e.second;
          std::sort(factors.begin(), factors.end(), [](Factor &f1, Factor &f2) {
            auto t1 = f1.first;
            auto t2 = f2.first;
            if (t1 != t2) {
              return compareTensorOrder(t1, t2);
            } else {
              return f1.second < f2.second;
            }
          });
        }
        // compare and insert
        Value res = addOp.getResult();
        int64_t dim = axisFactors.size();
        for (int64_t axis1 = 0; axis1 < dim; axis1++) {
          for (int64_t axis2 = axis1 + 1; axis2 < dim; axis2++) {
            if (axisFactors[axis1] == axisFactors[axis2]) {
              symmInfos[res].emplace_back(SymmAxisPair(axis1, axis2));
            }
          }
        }
      }
    }
    for (auto &e : symmInfos) {
      e.first.dump();
      llvm::errs() << e.second.size() << "\n";
      for (auto &pair : e.second) {
        llvm::errs() << "(" << pair.first << ", " << pair.second << ")\n";
      }
      llvm::errs() << "\n";
    }
  }
};

std::unique_ptr<Pass> mlir::tc::createPrintSymmetryPass() {
  return std::make_unique<PrintSymmetryPass>();
}
