#include "TCC/Dialect/TC/IR/TCOps.h"
#include "TCC/Dialect/TC/Transforms/Passes.h"
#include "TCC/Dialect/TC/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "llvm/Support/Debug.h"

namespace mlir::tc {
#define GEN_PASS_DEF_ADDCONTRACTMULTREEANALYSIS
#include "TCC/Dialect/TC/Transforms/Passes.h.inc"
} // namespace mlir::tc

#define DEBUG_TYPE "tree-analysis"

using namespace mlir;
using namespace mlir::tc;

using ValueWithAxesMap = DenseMap<Value, AxesTy>;
using Axis2ValueWithAxesMapTy = DenseMap<int64_t, ValueWithAxesMap>;
using AxisMapTy = Axis2ValueWithAxesMapTy;

inline void insertValueAxesToAxisMap(AxisMapTy &map, Value input,
                                     AxesMapTy &axesMap1, AxesMapTy &axesMap2,
                                     AxesMapTy &axesMap3) {
  for (auto [axis4, axis3] : llvm::enumerate(axesMap3)) {
    auto axis2 = axesMap2[axis3];
    if (axis2 >= 0) {
      auto axis1 = axesMap1[axis2];
      map[axis1][input].emplace_back(axis4);
    }
  }
}

inline void insertValueAxesToAxisMap(AxisMapTy &map, MulOp mulOp,
                                     AxesMapTy &axesMap1, AxesMapTy &axesMap2) {
  auto inputs3 = mulOp.getInputs();
  auto axesMaps3 = getAxesMaps(mulOp);
  for (auto [input3, axesMap3] : llvm::zip(inputs3, axesMaps3)) {
    insertValueAxesToAxisMap(map, input3, axesMap1, axesMap2, axesMap3);
  }
}

void TreeAnalysis(AddOp op) {
  if (!isElementWiseAddOp(op)) {
    return;
  }
  AxisMapTy map1;
  auto rank = getTensorRank(op.getResult());
  auto inputs = op.getInputs();
  auto axesMaps = getAxesMaps(op);
  for (auto [input, axesMap] : llvm::zip(inputs, axesMaps)) {
    if (auto contractOp = input.getDefiningOp<ContractOp>()) {
      auto input2 = contractOp.getInput();
      auto rank = getTensorRank(input2);
      auto axes2 = contractOp.getAxesArray();
      auto i2oAxisMap = getContractOpInput2OutputAxisMap(axes2, rank);
      if (auto mulOp = input2.getDefiningOp<MulOp>()) {
        insertValueAxesToAxisMap(map1, mulOp, axesMap, i2oAxisMap);
      } else {
        auto axesMap3 = getOrderedAxesMap(i2oAxisMap.size());
        insertValueAxesToAxisMap(map1, input2, axesMap, i2oAxisMap, axesMap3);
      }
    } else if (auto mulOp = input.getDefiningOp<MulOp>()) {
      auto i2oAxisMap = getOrderedAxesMap(axesMap.size());
      insertValueAxesToAxisMap(map1, mulOp, axesMap, i2oAxisMap);
    } else {
      auto i2oAxisMap = getOrderedAxesMap(axesMap.size());
      auto axesMap3 = getOrderedAxesMap(axesMap.size());
      insertValueAxesToAxisMap(map1, input, axesMap, i2oAxisMap, axesMap3);
    }
  }
  //   for (auto axis0 : llvm::seq<int64_t>(0, rank)) {
  //     llvm::errs() << axis0 << "\n";
  //     for (auto [input, axes] : map1[axis0]) {
  //       input.dump();
  //       for (auto axis3 : DenseSet<int64_t>(axes.begin(), axes.end())) {
  //         llvm::errs() << axis3 << ", ";
  //       }
  //       llvm::errs() << "\n";
  //     }
  //     llvm::errs() << "\n";
  //   }

  using ValueUsageTy = std::tuple<bool, int64_t, int64_t, AxesMapTy>;
  using ValueUsagesTy = SmallVector<ValueUsageTy>;
  using Value2ValueUsagesMap = DenseMap<Value, ValueUsagesTy>;
  using ValueMap = Value2ValueUsagesMap;
  ValueMap map2;
  for (auto [pos1, input] : llvm::enumerate(inputs)) {
    auto &axesMap = axesMaps[pos1];
    if (auto contractOp = input.getDefiningOp<ContractOp>()) {
      auto input2 = contractOp.getInput();
      auto rank = getTensorRank(input2);
      auto axes2 = contractOp.getAxesArray();
      auto axesMap2 = getContractOpInput2OutputAxisMap(axes2, rank);
      if (auto mulOp = input2.getDefiningOp<MulOp>()) {
        auto inputs3 = mulOp.getInputs();
        auto axesMaps3 = getAxesMaps(mulOp);
        for (auto [pos3, input3] : llvm::enumerate(inputs3)) {
          auto &axesMap3 = axesMaps3[pos3];
          ValueUsageTy usage(true, pos1, pos3, AxesMapTy());
          for (auto axis3 : axesMap3) {
            auto axis2 = axesMap2[axis3];
            if (axis2 >= 0) {
              auto axis1 = axesMap[axis2];
              std::get<3>(usage).emplace_back(axis1);
            } else {
              std::get<3>(usage).emplace_back(axis2);
            }
          }
          map2[input3].emplace_back(usage);
        }
      } else {
        auto axesMap3 = getOrderedAxesMap(axesMap2.size());
        ValueUsageTy usage(true, pos1, -1, AxesMapTy());
        for (auto axis3 : axesMap3) {
          auto axis2 = axesMap2[axis3];
          if (axis2 >= 0) {
            auto axis1 = axesMap[axis2];
            std::get<3>(usage).emplace_back(axis1);
          } else {
            std::get<3>(usage).emplace_back(axis2);
          }
        }
        map2[input2].emplace_back(usage);
      }
    } else if (auto mulOp = input.getDefiningOp<MulOp>()) {
      auto axesMap2 = getOrderedAxesMap(axesMap.size());
      auto inputs3 = mulOp.getInputs();
      auto axesMaps3 = getAxesMaps(mulOp);
      for (auto [pos3, input3] : llvm::enumerate(inputs3)) {
        auto &axesMap3 = axesMaps3[pos3];
        ValueUsageTy usage(false, pos1, pos3, AxesMapTy());
        for (auto axis3 : axesMap3) {
          auto axis2 = axesMap2[axis3];
          if (axis2 >= 0) {
            auto axis1 = axesMap[axis2];
            std::get<3>(usage).emplace_back(axis1);
          } else {
            std::get<3>(usage).emplace_back(axis2);
          }
        }
        map2[input3].emplace_back(usage);
      }
    } else {
      auto axesMap2 = getOrderedAxesMap(axesMap.size());
      auto axesMap3 = getOrderedAxesMap(axesMap.size());
      ValueUsageTy usage(false, pos1, -1, AxesMapTy());
      for (auto axis3 : axesMap3) {
        auto axis2 = axesMap2[axis3];
        if (axis2 >= 0) {
          auto axis1 = axesMap[axis2];
          std::get<3>(usage).emplace_back(axis1);
        } else {
          std::get<3>(usage).emplace_back(axis2);
        }
      }
      map2[input].emplace_back(usage);
    }
  }
  //   for (auto &e : map2) {
  //     e.first.dump();
  //     for (auto &usage : e.second) {
  //       llvm::errs() << "[";
  //       llvm::errs() << std::get<1>(usage) << ", ";
  //       llvm::errs() << std::get<0>(usage) << ", ";
  //       llvm::errs() << std::get<2>(usage) << "]\t";
  //       for (auto axis : std::get<3>(usage)) {
  //         if (axis >= 0) {
  //             llvm::errs() << axis << ", ";
  //         } else {
  //             llvm::errs() << "*" << -(axis + 1) << ", ";
  //         }
  //       }
  //       llvm::errs() << "\n";
  //     }
  //     llvm::errs() << "\n";
  //   }
  using MergeInfoTy = llvm::MapVector<int64_t, I64ArrayTy>;
  using SizeID2MergeInfoMapTy = DenseMap<int64_t, MergeInfoTy>;
  using SizeIDMapTy = SizeID2MergeInfoMapTy;
  SizeIDMapTy map3;
  for (auto [pos, input] : llvm::enumerate(inputs)) {
    if (auto contractOp = input.getDefiningOp<ContractOp>()) {
      auto axes = contractOp.getAxesArray();
      auto sizeIDs = getTensorSizeIDs(contractOp.getInput());
      for (auto [idx, axis] : llvm::enumerate(axes)) {
        auto sizeID = sizeIDs[axis];
        map3[sizeID][pos].emplace_back(idx);
      }
    }
  }
  op.dump();
  //   for (auto &e : map3) {
  //     auto &sizeID = e.first;
  //     auto &info = e.second;
  //     llvm::errs() << sizeID << "\n";
  //     for (auto [pos, axesPos] : info) {
  //       llvm::errs() << "  " << pos << ":  ";
  //       for (auto axisPos : axesPos) {
  //         llvm::errs() << axisPos << ", ";
  //       }
  //       llvm::errs() << "\n";
  //     }
  //     llvm::errs() << "\n";
  //   }
}

using InputPosTy = int64_t;
using PositionTy = std::tuple<InputPosTy, bool, InputPosTy>;
using UsageTy = std::pair<Value, AxesMapTy>;
using UsageMapTy = llvm::MapVector<PositionTy, UsageTy>;

template <> struct llvm::DenseMapInfo<PositionTy> {
  static inline PositionTy getEmptyKey() { return PositionTy(-1, false, -1); }
  static inline PositionTy getTombstoneKey() {
    return PositionTy(-2, false, -2);
  }
  static unsigned getHashValue(const PositionTy &Val) {
    return hash_combine(std::get<0>(Val), std::get<1>(Val), std::get<2>(Val));
  }
  static bool isEqual(const PositionTy &LHS, const PositionTy &RHS) {
    return LHS == RHS;
  }
};

namespace {
auto getUsage(Value input3, AxesMapTy &axesMap1, AxesMapTy &axesMap2,
              AxesMapTy &axesMap3) {
  UsageTy usage;
  usage.first = input3;
  for (auto axis3 : axesMap3) {
    auto axis2 = axesMap2[axis3];
    if (axis2 >= 0) {
      auto axis1 = axesMap1[axis2];
      std::get<1>(usage).emplace_back(axis1);
    } else {
      std::get<1>(usage).emplace_back(axis2);
    }
  }
  return usage;
}

void insertMulOpValueUsage(UsageMapTy &map, PositionTy pos, MulOp op,
                           AxesMapTy &axesMap1, AxesMapTy &axesMap2) {
  auto inputs3 = op.getInputs();
  auto axesMaps3 = getAxesMaps(op);
  for (auto [pos3, z3] : llvm::enumerate(llvm::zip(inputs3, axesMaps3))) {
    auto [input3, axesMap3] = z3;
    std::get<2>(pos) = pos3;
    auto usage = getUsage(input3, axesMap1, axesMap2, axesMap3);
    map.insert(std::pair(pos, usage));
  }
}

void TreeAnalysis2(AddOp op) {
  assert(isElementWiseAddOp(op));
  auto inputs = op.getInputs();
  auto axesMaps1 = getAxesMaps(op);
  UsageMapTy usageMap;
  for (auto [pos1, z1] : llvm::enumerate(llvm::zip(inputs, axesMaps1))) {
    auto [input1, axesMap1] = z1;
    if (auto contractOp = input1.getDefiningOp<ContractOp>()) {
      auto input2 = contractOp.getInput();
      auto rank2 = getTensorRank(input2);
      auto axes2 = contractOp.getAxesArray();
      auto axesMap2 = getContractOpInput2OutputAxisMap(axes2, rank2);
      if (auto mulOp = input2.getDefiningOp<MulOp>()) {
        PositionTy pos(pos1, true, -1);
        insertMulOpValueUsage(usageMap, pos, mulOp, axesMap1, axesMap2);
      } else {
        auto axesMap3 = getOrderedAxesMap(axesMap2.size());
        PositionTy pos(pos1, true, -1);
        auto usage = getUsage(input2, axesMap1, axesMap2, axesMap3);
        usageMap.insert(std::pair(pos, usage));
      }
    } else if (auto mulOp = input1.getDefiningOp<MulOp>()) {
      auto axesMap2 = getOrderedAxesMap(axesMap1.size());
      PositionTy pos(pos1, false, -1);
      insertMulOpValueUsage(usageMap, pos, mulOp, axesMap1, axesMap2);
    } else {
      PositionTy pos(pos1, false, -1);
      UsageTy usage(input1, axesMap1);
      usageMap.insert(std::pair(pos, usage));
    }
  }
  //   for (auto &[pos, usage] : usageMap) {
  //     auto [pos1, pos2, pos3] = pos;
  //     if (pos1 >= 0) {
  //       llvm::errs() << pos1 << ", ";
  //     } else {
  //       llvm::errs() << "x, ";
  //     }
  //     if (pos2) {
  //       llvm::errs() << "1, ";
  //     } else {
  //       llvm::errs() << "x, ";
  //     }
  //     if (pos3 >= 0) {
  //       llvm::errs() << pos3 << ":\n";
  //     } else {
  //       llvm::errs() << "x:\n";
  //     }
  //     llvm::errs() << "  ";
  //     for (auto axis : usage.second) {
  //       if (axis >= 0) {
  //         llvm::errs() << " " << axis << " ";
  //       } else {
  //         llvm::errs() << "*" << -(axis + 1) << " ";
  //       }
  //     }
  //     llvm::errs() << "\n";
  //     llvm::errs() << "  ";
  //     usage.first.dump();
  //   }
  llvm::SetVector<ContractOp> ctInputs;
  llvm::MapVector<ContractOp, llvm::MapVector<int64_t, SmallVector<int64_t>>>
      map4;
  for (auto [pos1, input1] : llvm::enumerate(inputs)) {
    auto ctOp = input1.getDefiningOp<ContractOp>();
    if (!ctOp || ctInputs.count(ctOp)) {
      continue;
    }
    auto input2 = ctOp.getInput();
    auto mulOp = input2.getDefiningOp<MulOp>();
    if (!mulOp) {
      continue;
    }
    ctInputs.insert(ctOp);
    auto inputs3 = mulOp.getInputs();
    auto axes2 = ctOp.getAxesArray();
    for (auto axis3 : axes2) {
      map4[ctOp][axis3] = SmallVector<int64_t>();
    }
    auto axesMaps3 = getAxesMaps(mulOp);
    for (auto [pos3, z3] : llvm::enumerate(llvm::zip(inputs3, axesMaps3))) {
      auto [input3, axesMap3] = z3;
      auto &axesMap0 = usageMap[PositionTy(pos1, true, pos3)].second;
      for (auto axis0 : axesMap0) {
        if (axis0 < 0) {
          axis0 = -(axis0 + 1);
          auto axis3 = axes2[axis0];
          map4[ctOp][axis3].emplace_back(pos3);
        }
      }
    }
  }
#if 0
  for (auto &[ctOp, map5] : map4) {
    ctOp.getInput().dump();
    for (auto &[axis3, vec] : map5) {
      llvm::errs() << axis3 << ": ";
      for (auto pos3 : vec) {
        llvm::errs() << pos3 << " ";
      }
      llvm::errs() << "\n";
    }
    llvm::errs() << "\n";
  }
#endif
  for (auto ctOp : ctInputs) {
    ctOp.getInput().dump();
    auto axisPosMap = map4[ctOp];
    auto axes = ctOp.getAxesArray();
    llvm::SetVector<int64_t> axisSet(axes.begin(), axes.end());
    AxesTy sortedAxes;
    llvm::SetVector<int64_t> resultAxes;
    while (!axisSet.empty()) {
      auto axis = axisSet.back();
      for (auto axis2 : axisSet) {
        if (axis == axis2) {
          continue;
        }
        auto inputPos1 = axisPosMap[axis];
        auto inputPos2 = axisPosMap[axis2];
        bool allIn = true;
        for (auto pos1 : inputPos1) {
          if (llvm::find(inputPos2, pos1) == inputPos2.end()) {
            allIn = false;
            break;
          }
        }
        if (allIn) {
          axis = axis2;
        }
      }
      axisSet.erase(llvm::find(axisSet, axis));
      sortedAxes.emplace_back(axis);
    }
    auto mulOp = ctOp.getInput().getDefiningOp<MulOp>();
    auto axesMaps3 = getAxesMaps(mulOp);
    auto numInputs = mulOp.getNumOperands();
    SmallVector<bool> notUsed(numInputs, true);
    for (auto axis : llvm::reverse(sortedAxes)) {
      llvm::errs() << axis << ": ";
      SmallVector<int64_t> usedPos;
      for (auto pos : axisPosMap[axis]) {
        if (notUsed[pos]) {
          notUsed[pos] = false;
          llvm::errs() << pos << " ";
          usedPos.emplace_back(pos);
        }
      }
      if (resultAxes.count(axis)) {
        llvm::errs() << "x ";
      }
      llvm::SetVector<int64_t> newResultAxes;
      for (auto pos : usedPos) {
        for (auto axis : axesMaps3[pos]) {
          newResultAxes.insert(axis);
        }
      }
      if (resultAxes.count(axis)) {
        for (auto axis : resultAxes) {
          newResultAxes.insert(axis);
        }
      }
      newResultAxes.erase(llvm::find(newResultAxes, axis));
      llvm::errs() << "-> (";
      for (auto axis : newResultAxes) {
        llvm::errs() << axis << ", ";
      }
      llvm::errs() << ")";
      llvm::errs() << "\n";
      for (auto axis : newResultAxes) {
        resultAxes.insert(axis);
      }
      auto f = llvm::find(resultAxes, axis);
      if (f != resultAxes.end())
        resultAxes.erase(f);
    }
    llvm::errs() << "\n";
  }
}
} // namespace

struct AddContractMulTreeAnalysisPass
    : public tc::impl::AddContractMulTreeAnalysisBase<
          AddContractMulTreeAnalysisPass> {
  void runOnOperation() override {
    auto funcOp = getOperation();
    auto &region = funcOp.getRegion();
    auto &entryBlock = *(region.getBlocks().begin());
    SmallVector<AddOp> workLists;
    for (auto &op : entryBlock) {
      if (auto addOp = dyn_cast_or_null<AddOp>(&op)) {
        if (isElementWiseAddOp(addOp)) {
          workLists.emplace_back(addOp);
        } else {
          LLVM_DEBUG(llvm::errs() << "Only support elementwise add.");
        }
      }
    }
    if (workLists.size() != 1) {
      // TODO: get from return op
      return;
    }
    TreeAnalysis2(workLists[0]);
  }
};

std::unique_ptr<Pass> mlir::tc::createAddContractMulTreeAnalysisPass() {
  return std::make_unique<AddContractMulTreeAnalysisPass>();
}
