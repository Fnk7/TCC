#include "TCC/Dialect/TC/IR/TCOps.h"
#include "TCC/Dialect/TC/Transforms/Passes.h"
#include "TCC/Dialect/TC/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "llvm/Support/Debug.h"

namespace mlir::tc {
#define GEN_PASS_DEF_SPLITCONTRACT
#include "TCC/Dialect/TC/Transforms/Passes.h.inc"
} // namespace mlir::tc

#define DEBUG_TYPE "split-contract"

using namespace mlir;
using namespace mlir::tc;

void splitContractOpWithMulOp(ContractOp op) {
  auto input = op.getInput();
  auto axes = getAxes(op);
  auto mulOp = input.getDefiningOp<MulOp>();
  if (axes.size() < 2 || !mulOp) {
    return;
  }
  OpBuilder builder(op);
  std::sort(axes.begin(), axes.end());
  for (auto axis : llvm::reverse(axes)) {
    AxesTy newAxes(1, axis);
    auto newOp = builder.create<ContractOp>(op.getLoc(), input, newAxes);
    input = newOp.getResult();
  }
  op.replaceAllUsesWith(input);
  op.erase();
}

template <> struct llvm::DenseMapInfo<I64ArrayTy> {
  static inline I64ArrayTy getEmptyKey() { return I64ArrayTy(); }
  static inline I64ArrayTy getTombstoneKey() { return I64ArrayTy(); }
  static unsigned getHashValue(const I64ArrayTy &Val) {
    return llvm::hash_combine_range(Val.begin(), Val.end());
  }
  static bool isEqual(const I64ArrayTy &LHS, const I64ArrayTy &RHS) {
    return LHS == RHS;
  }
};

namespace {
using InputPosSetTy = llvm::SetVector<int64_t>;
using Axis2InputSetMapTy = DenseMap<int64_t, InputPosSetTy>;
using AxisPairTy = std::pair<int64_t, int64_t>;
using AxesRelationTy = DenseMap<AxisPairTy, bool>;
// return true if set1 is subset of set2
bool isSubsetOf(InputPosSetTy &set1, InputPosSetTy &set2) {
  for (auto pos1 : set1) {
    if (llvm::find(set2, pos1) == set2.end()) {
      return false;
    }
  }
  return true;
}
// return true if set1 intersects with set2
bool isIntersecting(InputPosSetTy &set1, InputPosSetTy &set2) {
  for (auto pos1 : set1) {
    if (llvm::find(set2, pos1) != set2.end()) {
      return true;
    }
  }
  return false;
}

AxesRelationTy getSubsetRelation(AxesTy &axes, Axis2InputSetMapTy &map) {
  AxesRelationTy rel;
  for (auto axis1 : axes) {
    for (auto axis2 : axes) {
      AxisPairTy pair(axis1, axis2);
      rel[pair] = isSubsetOf(map[axis1], map[axis2]);
    }
  }
  return rel;
}

AxesRelationTy getIntersectRelation(AxesTy &axes, Axis2InputSetMapTy &map) {
  AxesRelationTy rel;
  for (auto axis1 : axes) {
    for (auto axis2 : axes) {
      AxisPairTy pair(axis1, axis2);
      rel[pair] = isIntersecting(map[axis1], map[axis2]);
    }
  }
  return rel;
}

// Point: selection chain, outer->inner:y, inner->outer:n
class SelectionNode;
using SelectionMapTy = DenseMap<AxesTy, std::unique_ptr<SelectionNode>>;
class SelectionNode {
  using SelectionsTy = SmallVector<AxesTy>;
  using SelectionNodesTy = SmallVector<SelectionNode *>;
  // a group of selections, ref.ed by position
  // the selection in the same group can be parallel
  using SelectionGroupTy = SmallVector<int64_t>;
  // all groups of selections
  using SelectionGroupsTy = SmallVector<SelectionGroupTy>;

public:
  SelectionsTy selections;
  SelectionNodesTy nodes;
  SelectionGroupsTy groups;

public:
  SelectionNode(AxesTy &axes, SelectionMapTy &map, AxesRelationTy &intersectRel,
                AxesRelationTy &subsetRel) {
    // pair.first  : not intersect group
    // pair.second : equality group, -1 represent to be a subset
    using GroupInfoTy = std::pair<int64_t, int64_t>;
    using GroupInfosTy = SmallVector<GroupInfoTy>;
    GroupInfosTy groupInfos;
    for (auto group : llvm::seq<int64_t>(0, axes.size())) {
      groupInfos.emplace_back(GroupInfoTy(group, group));
    }
    for (auto pos1 : llvm::seq<int64_t>(0, axes.size())) {
      for (auto pos2 : llvm::seq<int64_t>(pos1 + 1, axes.size())) {
        auto axis1 = axes[pos1];
        auto axis2 = axes[pos2];
        AxisPairTy pair1(axis1, axis2);
        if (intersectRel[pair1]) {
          auto &intersect1 = groupInfos[pos1].first;
          auto &intersect2 = groupInfos[pos2].first;
          auto &equality1 = groupInfos[pos1].second;
          auto &equality2 = groupInfos[pos2].second;
          auto minIntersect = std::min(intersect1, intersect2);
          intersect1 = minIntersect;
          intersect2 = minIntersect;
          AxisPairTy pair2(axis2, axis1);
          auto sub1 = subsetRel[pair1];
          auto sub2 = subsetRel[pair2];
          if (sub1 && sub2) {
            auto minEquality = std::min(equality1, equality2);
            equality1 = minEquality;
            equality2 = minEquality;
          } else if (sub1) {
            equality1 = -1;
          } else if (sub2) {
            equality2 = -1;
          }
        }
      }
    }
    llvm::MapVector<int64_t, llvm::SetVector<int64_t>> groupMap;
    for (auto [pos, info] : llvm::enumerate(groupInfos)) {
      auto &iGroup = info.first;
      while (iGroup != groupInfos[iGroup].first) {
        iGroup = groupInfos[iGroup].first;
      }
      auto &eGroup = info.second;
      while (eGroup >= 0 && eGroup != groupInfos[eGroup].second) {
        eGroup = groupInfos[eGroup].second;
      }
      if (eGroup >= 0) {
        groupMap[iGroup].insert(eGroup);
      }
    }
    for (auto &[iGroup, eGroups] : groupMap) {
      SelectionGroupTy group;
      for (auto eGroup : eGroups) {
        AxesTy toSelectAxes;
        AxesTy selectedAxes;
        for (auto [axis, info] : llvm::zip(axes, groupInfos)) {
          if (info.first != iGroup) {
            continue;
          }
          if (info.second != eGroup) {
            toSelectAxes.emplace_back(axis);
          } else {
            selectedAxes.emplace_back(axis);
          }
        }
        selections.emplace_back(selectedAxes);
        if (toSelectAxes.size()) {
          if (!map.count(toSelectAxes)) {
            auto node = std::make_unique<SelectionNode>(
                toSelectAxes, map, intersectRel, subsetRel);
            map.insert(std::pair(toSelectAxes, std::move(node)));
          }
          auto *node = map[toSelectAxes].get();
          nodes.emplace_back(node);
        } else {
          nodes.emplace_back(nullptr);
        }
        group.emplace_back(nodes.size() - 1);
      }
      groups.emplace_back(group);
    }
  }

  int64_t getPossibles() {
    int64_t res1 = 1;
    for (auto group : groups) {
      int64_t res2 = 0;
      for (auto pos : group) {
        if (nodes[pos]) {
          res2 += nodes[pos]->getPossibles();
        } else {
          res2 += 1;
        }
      }
      res1 *= res2;
    }
    return res1;
  }

  void dump(int indent) {
    for (auto [idx1, group] : llvm::enumerate(groups)) {
      for (auto [idx2, pos] : llvm::enumerate(group)) {
        if (idx1 != 0 || idx2 != 0) {
          llvm::errs().indent(indent);
        }
        std::string toPrint = " ";
        if (groups.size() > 1) {
          if (idx2 == 0) {
            toPrint += "*";
          } else {
            toPrint += "*";
          }
        }
        toPrint += "-> ";
        auto &selection = selections[pos];
        if (selection.size() != 1) {
          toPrint += "(";
        }
        for (auto [idx3, axis] : llvm::enumerate(selection)) {
          if (idx3 != 0) {
            toPrint += ", ";
          }
          toPrint += std::to_string(axis);
        }
        if (selection.size() != 1) {
          toPrint += ")";
        }
        llvm::errs() << toPrint;
        if (auto *node = nodes[pos]) {
          node->dump(indent + toPrint.size());
        } else {
          llvm::errs() << "\n";
        }
      }
    }
  }
  void dump() {
    llvm::errs() << "Possible chains number: " << getPossibles() << "\n";
    dump(0);
  }
};

Axis2InputSetMapTy getAxis2InputSetMap(AxesTy &axes,
                                       AxesMapsTy &inputAxesMaps) {
  Axis2InputSetMapTy map;
  for (auto [pos, axesMap] : llvm::enumerate(inputAxesMaps)) {
    for (auto axis : axes) {
      if (llvm::find(axesMap, axis) != axesMap.end()) {
        map[axis].insert(pos);
      }
    }
  }
  return map;
}

SelectionMapTy analysisContractMul(AxesTy &axes, Axis2InputSetMapTy &map) {
  AxesRelationTy intersectRel = getIntersectRelation(axes, map);
  AxesRelationTy subsetRel = getSubsetRelation(axes, map);
  SelectionMapTy selectionMap;
  auto node = std::make_unique<SelectionNode>(axes, selectionMap, intersectRel,
                                              subsetRel);
  selectionMap.insert(std::pair(axes, std::move(node)));
  return selectionMap;
}

int64_t getRankFromAxesMaps(AxesMapsTy &axesMaps) {
  int64_t maxAxis = -1;
  for (auto &axesMap : axesMaps) {
    for (auto axis : axesMap) {
      maxAxis = std::max(axis, maxAxis);
    }
  }
  return maxAxis + 1;
}

void mapOriginAxis2NewAxis(AxesMapTy &axesMap) {
  int64_t newAxis = 0;
  for (auto &axis : axesMap) {
    if (axis >= 0) {
      axis = newAxis++;
    }
  }
}

AxesMapTy getOrigin2NewAxesMap(AxesMapsTy &axesMaps) {
  auto rank = getRankFromAxesMaps(axesMaps);
  AxesMapTy origin2NewMap(rank, -1);
  for (auto &axesMap : axesMaps) {
    for (auto axis : axesMap) {
      origin2NewMap[axis] = axis;
    }
  }
  mapOriginAxis2NewAxis(origin2NewMap);
  return origin2NewMap;
}

AxesMapTy getNew2OriginAxesMap(AxesMapTy &origin2NewMap) {
  AxesMapTy new2OriginMap;
  for (auto [originAxis, newAxis] : llvm::enumerate(origin2NewMap)) {
    if (newAxis >= 0) {
      new2OriginMap.emplace_back(originAxis);
    }
  }
  return new2OriginMap;
}

void applyUpdateAxesMapWithAxes(AxesMapTy &axesMap1, AxesTy &axes) {
  for (auto &axis : axes) {
    auto &axis2 = axesMap1[axis];
    axis = axis2;
    axis2 = -1;
  }
  mapOriginAxis2NewAxis(axesMap1);
}

void applyAxesMap(AxesMapTy &axesMap1, AxesMapTy &axesMap2) {
  for (auto &axis : axesMap2) {
    axis = axesMap1[axis];
  }
}

void applyAxesMap(AxesMapTy &axesMap1, AxesMapsTy &axesMaps) {
  for (auto &axesMap2 : axesMaps) {
    applyAxesMap(axesMap1, axesMap2);
  }
}

std::pair<Value, AxesMapTy> createContractMulWithSelectedAxes(
    ContractOp ctOp, AxesTy &selectedAxes, Axis2InputSetMapTy &inputSetMap,
    SmallVector<Value> &originInputs, AxesMapsTy &originAxesMaps,
    SmallVector<Value> &mulInputs, AxesMapsTy &mulAxesMaps) {
  // collect inputs for MulOp
  for (auto axis : selectedAxes) {
    for (auto pos : inputSetMap[axis]) {
      if (auto input = originInputs[pos]) {
        originInputs[pos] = Value(nullptr);
        mulInputs.emplace_back(input);
        mulAxesMaps.emplace_back(originAxesMaps[pos]);
      }
    }
  }
  // origin mul axis -> new mul axis map
  AxesMapTy origin2NewMap = getOrigin2NewAxesMap(mulAxesMaps);
  // create MulOp if # inputs > 1
  OpBuilder builder(ctOp);
  Value result;
  if (mulInputs.size() >= 2) {
    applyAxesMap(origin2NewMap, mulAxesMaps);
    auto mulOp = builder.create<MulOp>(ctOp.getLoc(), mulInputs, mulAxesMaps);
    result = mulOp.getResult();
  } else {
    assert(mulInputs.size() == 1);
    result = mulInputs[0];
  }
  // create ContractOp if # selectedAxes >= 1
  if (selectedAxes.size()) {
    applyUpdateAxesMapWithAxes(origin2NewMap, selectedAxes);
    auto ctOp2 =
        builder.create<ContractOp>(ctOp.getLoc(), result, selectedAxes);
    result = ctOp2.getResult();
  } else {
    assert(result);
  }
  auto retAxesMap = getNew2OriginAxesMap(origin2NewMap);
  return std::pair(result, retAxesMap);
}

std::pair<SmallVector<Value>, AxesMapsTy>
getContractMulOfSelection(ContractOp ctOp, SelectionNode *node,
                          Axis2InputSetMapTy &inputSetMap,
                          SmallVector<Value> &inputs, AxesMapsTy &axesMaps) {
  // empty selection
  if (!node)
    return std::pair(SmallVector<Value>(), AxesMapsTy());
  SmallVector<Value> retInputs;
  AxesMapsTy retAxesMaps;
  for (auto &group : node->groups) {
    // pick one from one group
    auto pos = group[0];
    auto [mulInputs, mulAxesMaps] = getContractMulOfSelection(
        ctOp, node->nodes[pos], inputSetMap, inputs, axesMaps);
    auto selectedAxes = node->selections[pos];
    auto [result, retAxesMap] = createContractMulWithSelectedAxes(
        ctOp, selectedAxes, inputSetMap, inputs, axesMaps, mulInputs,
        mulAxesMaps);
    retInputs.emplace_back(result);
    retAxesMaps.emplace_back(retAxesMap);
  }
  return std::pair(retInputs, retAxesMaps);
}

Value getContractMulOfRootSelection(ContractOp ctOp, SelectionNode *node,
                                    Axis2InputSetMapTy &inputSetMap,
                                    SmallVector<Value> &inputs,
                                    AxesMapsTy &axesMaps) {
  auto [mulInputs, mulAxesMaps] =
      getContractMulOfSelection(ctOp, node, inputSetMap, inputs, axesMaps);
  AxesTy selectedAxes;
  auto [result, retAxesMap] =
      createContractMulWithSelectedAxes(ctOp, selectedAxes, inputSetMap, inputs,
                                        axesMaps, mulInputs, mulAxesMaps);
  return result;
}
} // namespace

void dumpValueDefiningChain(Value v, llvm::SetVector<Operation *> &dumpped) {
  // Dump
  auto *op = v.getDefiningOp();
  if (!op || dumpped.count(op)) {
    return;
  }
  for (auto v : op->getOperands()) {
    dumpValueDefiningChain(v, dumpped);
  }
  dumpped.insert(op);
  op->dump();
}

void dumpValueDefiningChain(Value v) {
  llvm::SetVector<Operation *> dumpped;
  dumpValueDefiningChain(v, dumpped);
}

void splitContractAndLiftMul(ContractOp op2) {
  auto input2 = op2.getInput();
  auto op3 = input2.getDefiningOp<MulOp>();
  if (!op3) {
    return;
  }
  auto axes2 = getAxes(op2);
  SmallVector<Value> inputs(op3.getInputs());
  auto axesMaps3 = getAxesMaps(op3);
  Axis2InputSetMapTy inputSetMap = getAxis2InputSetMap(axes2, axesMaps3);
  auto selectionMap = analysisContractMul(axes2, inputSetMap);
  auto *node = selectionMap[axes2].get();
  // TODO:
  // currently, get the 0 chain
  if (axes2.size() <= 1) {
    return;
  }
  auto result =
      getContractMulOfRootSelection(op2, node, inputSetMap, inputs, axesMaps3);
  op3.dump();
  op2.dump();
  node->dump();
  dumpValueDefiningChain(result);
  llvm::errs() << "\n";
  op2.replaceAllUsesWith(result);
  op2.erase();
}

struct SplitContractPass
    : public tc::impl::SplitContractBase<SplitContractPass> {
  void runOnOperation() override {
    auto funcOp = getOperation();
    auto &region = funcOp.getRegion();
    auto &entryBlock = *(region.getBlocks().begin());
    llvm::SetVector<ContractOp> workLists;
    for (auto &op : entryBlock) {
      if (auto contractOp = dyn_cast_or_null<ContractOp>(&op)) {
        workLists.insert(contractOp);
      }
    }
    for (auto contractOp : workLists) {
      splitContractAndLiftMul(contractOp);
    }
  }
};

std::unique_ptr<Pass> mlir::tc::createSplitContractPass() {
  return std::make_unique<SplitContractPass>();
}
