#include "TCC/Dialect/TC/IR/TCOps.h"
#include "TCC/Dialect/TC/Transforms/Passes.h"
#include "TCC/Dialect/TC/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "llvm/Support/Debug.h"

namespace mlir::tc {
#define GEN_PASS_DEF_PRINTSYMMETRY
#include "TCC/Dialect/TC/Transforms/Passes.h.inc"
} // namespace mlir::tc

#define DEBUG_TYPE "print-symmetry"

// cond 1: 1/6
// a(i,j,k) = a(j,i,k)
// a(i,j,k) = a(i,k,j)
// => a(i,j,k) = a(j,i,k) = a(j,k,i) = a(k,j,i)
// cond 2: 1/3
// a(i,j,k) = a(k,i,j)
// => a(i,j,k) = a(j,k,i)

using namespace mlir;
using namespace mlir::tc;

using SymmAxisPair = std::pair<int64_t, int64_t>;
using SymmInfo = SmallVector<SymmAxisPair>;
using ValueSymmInfoMap = DenseMap<Value, SymmInfo>;

template <typename T>
void collectAbelianOpSymmInfo(ValueSymmInfoMap &symmInfos, T op) {
  auto inputs = op.getInputs();
  auto axesMaps = getAxesMaps(op);
  using Factor = std::pair<Value, int64_t>;
  using Factors = SmallVector<Factor>;
  using AxisFactorsMap = DenseMap<int64_t, Factors>;
  // map to factors
  AxisFactorsMap axisFactors;
  for (auto [input, axesMap] : llvm::zip(inputs, axesMaps)) {
    for (auto [iAxis, oAxis] : llvm::enumerate(axesMap)) {
      bool found = false;
      if (symmInfos.count(input)) {
        for (auto &pair : symmInfos[input]) {
          if (pair.first == iAxis || pair.second == iAxis) {
            found = true;
            axisFactors[oAxis].emplace_back(Factor(input, pair.first));
            axisFactors[oAxis].emplace_back(Factor(input, pair.second));
          }
        }
      }
      if (!found) {
        axisFactors[oAxis].emplace_back(Factor(input, iAxis));
      }
    }
  }
  // sort factors for later compare
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
  // compare and insert axis pair into symminfos
  Value res = op.getResult();
  int64_t dim = axisFactors.size();
  for (int64_t axis1 = 0; axis1 < dim; axis1++) {
    for (int64_t axis2 = axis1 + 1; axis2 < dim; axis2++) {
      if (axisFactors[axis1] == axisFactors[axis2]) {
        symmInfos[res].emplace_back(SymmAxisPair(axis1, axis2));
      }
    }
  }
}

void collectContractOpSymmInfo(ValueSymmInfoMap &symmInfos, ContractOp op) {
  auto input = op.getInput();
  if (!symmInfos.count(input))
    return;
  auto rank = getTensorRank(input);
  auto axes = getAxes(op);
  auto i2oAxisMap = getContractOpInput2OutputAxisMap(axes, rank);
  SymmInfo symmInfo;
  for (auto &pair : symmInfos[input]) {
    auto axis1 = i2oAxisMap[pair.first];
    auto axis2 = i2oAxisMap[pair.second];
    if (axis1 >= 0 && axis2 >= 0) {
      symmInfo.emplace_back(axis1, axis2);
    }
  }
  if (symmInfo.size()) {
    symmInfos[op.getResult()] = symmInfo;
  }
}

void collectPermuteOpSymmInfo(ValueSymmInfoMap &symmInfos, PermuteOp op) {
  auto input = op.getInput();
  if (!symmInfos.count(input))
    return;
  auto axesMap = getAxesMap(op);
  auto &symmInfo = symmInfos[op.getResult()];
  for (auto &pair : symmInfo) {
    pair.first = axesMap[pair.first];
    pair.second = axesMap[pair.second];
  }
}

struct PrintSymmetryPass
    : public tc::impl::PrintSymmetryBase<PrintSymmetryPass> {
  void runOnOperation() override {
    ValueSymmInfoMap symmInfos;
    auto funcOp = getOperation();
    auto &region = funcOp.getRegion();
    auto &entryBlock = *(region.getBlocks().begin());
    for (auto &op : entryBlock) {
      LLVM_DEBUG(llvm::dbgs() << "Inferring symmetry for: " << op << "\n");
      if (auto mulOp = dyn_cast_or_null<MulOp>(&op)) {
        collectAbelianOpSymmInfo(symmInfos, mulOp);
      } else if (auto addOp = dyn_cast_or_null<AddOp>(&op)) {
        collectAbelianOpSymmInfo(symmInfos, addOp);
      } else if (auto contractOp = dyn_cast_or_null<ContractOp>(&op)) {
        collectContractOpSymmInfo(symmInfos, contractOp);
      } else if (auto permuteOp = dyn_cast_or_null<PermuteOp>(&op)) {
        collectPermuteOpSymmInfo(symmInfos, permuteOp);
      } else if (auto negOp = dyn_cast_or_null<NegOp>(&op)) {
        if (symmInfos.count(negOp.getInput())) {
          symmInfos[negOp.getResult()] = symmInfos[negOp.getInput()];
        }
      }
    }
    for (auto &op : entryBlock) {
      auto v = op.getResult(0);
      if (symmInfos.count(v)) {
        op.dump();
        for (auto &pair : symmInfos[v]) {
          llvm::errs() << "(" << pair.first << ", " << pair.second << ")\n";
        }
        llvm::errs() << "\n";
      }
    }
  }
};

std::unique_ptr<Pass> mlir::tc::createPrintSymmetryPass() {
  return std::make_unique<PrintSymmetryPass>();
}
