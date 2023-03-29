#include "TCC/Dialect/TC/IR/TCOps.h"
#include "TCC/Dialect/TC/Utils/Utils.h"

#include "TCC/Dialect/TC/Transforms/Symmetry.h"

namespace mlir::tc::symmetry {
// Make SymmArray unique
void makeSymmetryArrayValid(SymmArrayTy &symmetry) {
  for (auto &pair : symmetry) {
    if (pair.first > pair.second) {
      std::swap(pair.first, pair.second);
    }
  }
  llvm::sort(symmetry, [](SymmPairTy &pair1, SymmPairTy &pair2) -> bool {
    return pair1.first < pair2.first;
  });
}

SymmArrayTy getResultSymmetryByAxesMaps(ValueRange inputs,
                                        AxesMapsRefTy axesMaps) {
  ValueSymmetryMapTy valueSymmetry;
  for (auto input : inputs) {
    valueSymmetry[input] = getTensorSymmetry(input);
  }
  using FactorTy = std::pair<Value, int64_t>;
  using FactorsTy = SmallVector<FactorTy>;
  using AxisFactorsMapTy = DenseMap<int64_t, FactorsTy>;
  // Map result axis to factors
  AxisFactorsMapTy axisFactorsMap;
  for (auto [input, axesMap] : llvm::zip(inputs, axesMaps)) {
    for (auto [idx, oAxis] : llvm::enumerate(axesMap)) {
      bool isSymm = false;
      AxisTy iAxis = idx;
      for (auto pair : valueSymmetry[input]) {
        if (pair.first == iAxis || pair.second == iAxis) {
          isSymm = true;
          axisFactorsMap[oAxis].push_back(FactorTy(input, pair.first));
          axisFactorsMap[oAxis].push_back(FactorTy(input, pair.second));
        }
      }
      if (!isSymm)
        axisFactorsMap[oAxis].emplace_back(FactorTy(input, iAxis));
    }
  }
  // Sort factors for later compare
  for (auto &[_, factors] : axisFactorsMap) {
    std::sort(factors.begin(), factors.end(), [](FactorTy &f1, FactorTy &f2) {
      auto t1 = f1.first;
      auto t2 = f2.first;
      if (t1 != t2) {
        return compareTensorOrder(t1, t2);
      } else {
        return f1.second < f2.second;
      }
    });
  }
  SymmArrayTy resultSymmetry;
  int64_t rank = axisFactorsMap.size();
  for (int64_t axis1 = 0; axis1 < rank; axis1++) {
    for (int64_t axis2 = axis1 + 1; axis2 < rank; axis2++) {
      if (axisFactorsMap[axis1] == axisFactorsMap[axis2]) {
        resultSymmetry.push_back(SymmPairTy(axis1, axis2));
      }
    }
  }
  makeSymmetryArrayValid(resultSymmetry);
  return resultSymmetry;
}
template <typename T>
void collectAbelianOpSymmInfo(ValueSymmetryMapTy &symmInfos, T op) {
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
        symmInfos[res].emplace_back(SymmPairTy(axis1, axis2));
      }
    }
  }
}
SymmArrayTy getResultSymmetryByAxes(Value input, AxesRefTy axes) {
  auto inputSymmetry = getTensorSymmetry(input);
  if (inputSymmetry.empty())
    return {};
  auto rank = getTensorRank(input);
  auto i2oAxesMap = getContractOpInput2OutputAxisMap(axes, rank);
  SymmArrayTy resultSymmetry;
  for (auto pair : inputSymmetry) {
    auto axis1 = i2oAxesMap[pair.first];
    auto axis2 = i2oAxesMap[pair.second];
    if (axis1 >= 0 && axis2 >= 0)
      resultSymmetry.emplace_back(axis1, axis2);
  }
  makeSymmetryArrayValid(resultSymmetry);
  return resultSymmetry;
}
void collectContractOpSymmInfo(ValueSymmetryMapTy &symmInfos, ContractOp op) {
  auto input = op.getInput();
  if (!symmInfos.count(input))
    return;
  auto rank = getTensorRank(input);
  auto axes = op.getAxesArray();
  auto i2oAxisMap = getContractOpInput2OutputAxisMap(axes, rank);
  SymmArrayTy symmInfo;
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
SymmArrayTy getResultSymmetryByAxesMap(Value input, AxesMapRefTy axesMap) {
  auto symmetry = getTensorSymmetry(input);
  for (auto &pair : symmetry) {
    pair.first = axesMap[pair.first];
    pair.second = axesMap[pair.second];
  }
  makeSymmetryArrayValid(symmetry);
  return symmetry;
}
void collectPermuteOpSymmInfo(ValueSymmetryMapTy &symmInfos, PermuteOp op) {
  auto input = op.getInput();
  if (!symmInfos.count(input))
    return;
  auto axesMap = op.getAxesMapArray();
  auto &symmInfo = symmInfos[op.getResult()];
  for (auto &pair : symmInfo) {
    pair.first = axesMap[pair.first];
    pair.second = axesMap[pair.second];
  }
}
void collectSymmInfosOfValue(ValueSymmetryMapTy &symmInfos, Value value) {
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
void cleanEmptyInfoInSymmInfos(ValueSymmetryMapTy &symmInfos) {
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
