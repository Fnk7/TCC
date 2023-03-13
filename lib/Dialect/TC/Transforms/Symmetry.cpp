#include "TCC/Dialect/TC/IR/TCOps.h"
#include "TCC/Dialect/TC/Utils/Utils.h"

#include "TCC/Dialect/TC/Transforms/Symmetry.h"

namespace mlir::tc::symmetry {
template <typename T>
void collectAbelianOpSymmInfo(ValueSymmInfoMapTy &symmInfos, T op) {
  auto inputs = op.getInputs();
  auto axesMaps = getAxesMaps(op);
  using Factor = std::pair<Value, int64_t>;
  using Factors = SmallVector<Factor>;
  using AxisFactorsMap = DenseMap<int64_t, Factors>;
  // map to factors
  AxisFactorsMap axisFactors;
  for (auto [input, axesMap] : llvm::zip(inputs, axesMaps)) {
    for (auto e : llvm::enumerate(axesMap)) {
      int64_t axis = e.index();
      bool found = false;
      if (symmInfos.count(input)) {
        for (auto &pair : symmInfos[input]) {
          if (pair.first == axis || pair.second == axis) {
            found = true;
            axisFactors[e.value()].emplace_back(Factor(input, pair.first));
            axisFactors[e.value()].emplace_back(Factor(input, pair.second));
          }
        }
      }
      if (!found) {
        axisFactors[e.value()].emplace_back(Factor(input, axis));
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
        symmInfos[res].emplace_back(SymmAxisPairTy(axis1, axis2));
      }
    }
  }
}
void collectContractOpSymmInfo(ValueSymmInfoMapTy &symmInfos, ContractOp op) {
  auto input = op.getInput();
  if (!symmInfos.count(input))
    return;
  auto rank = getTensorRank(input);
  auto axes = getAxes(op);
  auto i2oAxisMap = getContractOpInput2OutputAxisMap(axes, rank);
  SymmetryInfoTy symmInfo;
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
void collectPermuteOpSymmInfo(ValueSymmInfoMapTy &symmInfos, PermuteOp op) {
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
void collectSymmInfosOfValue(ValueSymmInfoMapTy &symmInfos, Value value) {
  auto op = value.getDefiningOp();
  if (!op) {
    return;
  }
  for (auto input : op->getOperands()) {
    collectSymmInfosOfValue(symmInfos, input);
  }
  if (auto op = value.getDefiningOp<AddOp>()) {
    collectAbelianOpSymmInfo(symmInfos, op);
  } else if (auto op = value.getDefiningOp<MulOp>()) {
    collectAbelianOpSymmInfo(symmInfos, op);
  } else if (auto op = value.getDefiningOp<ContractOp>()) {
    collectContractOpSymmInfo(symmInfos, op);
  }
}
void cleanEmptyInfoInSymmInfos(ValueSymmInfoMapTy &symmInfos) {
  SmallVector<Value> toErase;
  for (auto &[value, info] : symmInfos) {
    if (info.empty()) {
      toErase.emplace_back(value);
    }
  }
  for (auto value : toErase) {
    symmInfos.erase(value);
  }
}
} // namespace mlir::tc::symmetry
