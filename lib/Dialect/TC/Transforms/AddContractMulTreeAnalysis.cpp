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
      auto axes2 = getAxes(contractOp);
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
      auto axes2 = getAxes(contractOp);
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
      auto axes = getAxes(contractOp);
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
    TreeAnalysis(workLists[0]);
  }
};

std::unique_ptr<Pass> mlir::tc::createAddContractMulTreeAnalysisPass() {
  return std::make_unique<AddContractMulTreeAnalysisPass>();
}
