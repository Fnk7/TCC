#include "TCC/Dialect/TC/Transforms/ContractOrder.h"
#include "TCC/Dialect/TC/Utils/AxesMap.h"
#include "TCC/Dialect/TC/Utils/Utils.h"

#include "TCC/Utils.h"

#include "llvm/ADT/SetVector.h"

#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace mlir::tc;

namespace {
// Map sum axis to related input postions set
using InputPositionSetTy = llvm::SetVector<int64_t>;
using Axis2InputSetMapTy = DenseMap<int64_t, InputPositionSetTy>;
Axis2InputSetMapTy getAxis2InputSetMap(AxesRefTy axes, AxesMapsRefTy axesMaps) {
  Axis2InputSetMapTy axis2InputSetMap;
  for (auto [position, axesMap] : llvm::enumerate(axesMaps)) {
    for (auto axis : axes)
      if (llvm::find(axesMap, axis) != axesMap.end())
        axis2InputSetMap[axis].insert(position);
  }
  return axis2InputSetMap;
}

// Calculate sum (axis,axis) pair relation
using AxesRelationTy = DenseMap<AxisPairTy, bool>;
template <typename Func>
AxesRelationTy getAxesRelation(AxesRefTy axes, Func func) {
  AxesRelationTy relation;
  for (auto axis1 : axes) {
    for (auto axis2 : axes) {
      AxisPairTy pair(axis1, axis2);
      relation[pair] = func(axis1, axis2);
    }
  }
  return relation;
}
bool isIntersecting(InputPositionSetTy &set1, InputPositionSetTy &set2) {
  for (auto pos1 : set1)
    if (llvm::find(set2, pos1) != set2.end())
      return true;
  return false;
}
auto getAxesIntersectRelation(AxesRefTy axes, Axis2InputSetMapTy &map) {
  return getAxesRelation(axes, [&](AxisTy axis1, AxisTy axis2) -> bool {
    return isIntersecting(map[axis1], map[axis2]);
  });
}
bool isSubsetOf(InputPositionSetTy &set1, InputPositionSetTy &set2) {
  for (auto pos1 : set1)
    if (llvm::find(set2, pos1) == set2.end())
      return false;
  return true;
}
AxesRelationTy getAxesSubsetRelation(AxesRefTy axes, Axis2InputSetMapTy &map) {
  return getAxesRelation(axes, [&](AxisTy axis1, AxisTy axis2) -> bool {
    return isSubsetOf(map[axis1], map[axis2]);
  });
}

// Selection node
class SelectionNode;
using Axes2SelectionNodeMapTy =
    DenseMap<AxesTy, std::unique_ptr<SelectionNode>>;
class SelectionNode {
private:
  int64_t numberOfChains = 0;
  SmallVector<AxesTy> axesSelections;
  SmallVector<SelectionNode *> childs;

  SelectionNode() {}

public:
  using Group2AxesMap = DenseMap<int64_t, AxesTy>;
  static SelectionNode *getGroupNode(const AxesTy &axes,
                                     Axes2SelectionNodeMapTy &axes2NodeMap,
                                     AxesRelationTy &axesIntersectRel,
                                     AxesRelationTy &axesSubsetRel) {
    if (axes2NodeMap.count(axes))
      return axes2NodeMap[axes].get();
    // Map axis into intersect groups
    UnionFind intersectUnion(axes.size());
    for (auto pos1 : llvm::seq<int64_t>(0, axes.size())) {
      for (auto pos2 : llvm::seq<int64_t>(pos1 + 1, axes.size())) {
        AxisPairTy pair(axes[pos1], axes[pos2]);
        if (axesIntersectRel[pair])
          intersectUnion.Union(pos1, pos2);
      }
    }
    Group2AxesMap group2Axes;
    for (auto [pos, axis] : llvm::enumerate(axes))
      group2Axes[intersectUnion.find(pos)].emplace_back(axis);
    // If only 1 group, then not create group node
    if (group2Axes.size() <= 1)
      return getSelectionNode(axes, axes2NodeMap, axesIntersectRel,
                              axesSubsetRel);
    // Create group node
    auto *node = new SelectionNode();
    for (auto &[_, groupAxes] : group2Axes) {
      // Get node of axes in same group
      auto *child = getSelectionNode(groupAxes, axes2NodeMap, axesIntersectRel,
                                     axesSubsetRel);
      node->childs.emplace_back(child);
    }
    axes2NodeMap[axes] = std::unique_ptr<SelectionNode>(node);
    return node;
  }
  static SelectionNode *getSelectionNode(const AxesTy &axes,
                                         Axes2SelectionNodeMapTy &axes2NodeMap,
                                         AxesRelationTy &axesIntersectRel,
                                         AxesRelationTy &axesSubsetRel) {
    if (axes2NodeMap.count(axes))
      return axes2NodeMap[axes].get();
    // Map axis into equal groups
    UnionFind equalUnion(axes.size());
    SmallVector<bool> realSubset(axes.size(), false);
    for (auto pos1 : llvm::seq<int64_t>(0, axes.size())) {
      for (auto pos2 : llvm::seq<int64_t>(pos1 + 1, axes.size())) {
        AxisPairTy pair1(axes[pos1], axes[pos2]);
        AxisPairTy pair2(axes[pos2], axes[pos1]);
        auto sub1 = axesSubsetRel[pair1];
        auto sub2 = axesSubsetRel[pair2];
        if (sub1 && sub2)
          equalUnion.Union(pos1, pos2);
        else if (sub1 && !sub2)
          realSubset[pos1] = true;
        else if (!sub1 && sub2)
          realSubset[pos2] = true;
      }
    }
    Group2AxesMap group2Axes;
    for (auto [pos, axis] : llvm::enumerate(axes))
      if (realSubset[pos])
        group2Axes[-1].emplace_back(axis);
      else
        group2Axes[equalUnion.find(pos)].emplace_back(axis);
    // Create group node
    auto *node = new SelectionNode();
    for (auto &[group, groupAxes] : group2Axes) {
      if (group == -1)
        continue;
      AxesTy restAxes;
      for (auto &[group2, groupAxes2] : group2Axes)
        if (group != group2)
          restAxes.append(groupAxes2);
      auto *child =
          getGroupNode(restAxes, axes2NodeMap, axesIntersectRel, axesSubsetRel);
      node->axesSelections.emplace_back(groupAxes);
      node->childs.emplace_back(child);
    }
    axes2NodeMap[axes] = std::unique_ptr<SelectionNode>(node);
    return node;
  }

  int64_t getNumberOfChains() {
    if (numberOfChains == 0) {
      if (axesSelections.empty()) {
        // Group node
        numberOfChains = 1;
        for (auto child : childs)
          numberOfChains *= child->getNumberOfChains();
      } else {
        // Select node
        for (auto child : childs)
          numberOfChains += child->getNumberOfChains();
      }
    }
    return numberOfChains;
  }

  void dump(int indent = 0);
  void dumpOrder(int64_t orderID);
};
} // namespace

namespace mlir::tc {
class ContractOrders::Impl {
public:
  Axes2SelectionNodeMapTy axes2NodeMap;
  SelectionNode *rootNode;

  Impl(AxesRefTy axes, AxesMapsRefTy axesMaps) {
    auto axis2InputSetMap = getAxis2InputSetMap(axes, axesMaps);
    auto axesSubsetRel = getAxesSubsetRelation(axes, axis2InputSetMap);
    auto axesIntersectRel = getAxesIntersectRelation(axes, axis2InputSetMap);
    rootNode = SelectionNode::getGroupNode(AxesTy(axes), axes2NodeMap,
                                           axesIntersectRel, axesSubsetRel);
  }
};

// Constructor and destructor
ContractOrders::ContractOrders(AxesRefTy axes, AxesMapsRefTy axesMaps)
    : impl(new ContractOrders::Impl(axes, axesMaps)) {}
ContractOrders::~ContractOrders() { delete impl; }

int64_t ContractOrders::getNumberOfOrders() {
  return impl->rootNode->getNumberOfChains();
}

void ContractOrders::dump() { impl->rootNode->dump(); }
void ContractOrders::dumpOrders() {
  for (int64_t orderID = 0; orderID < getNumberOfOrders(); orderID++) {
    impl->rootNode->dumpOrder(orderID);
    llvm::errs() << "\n";
  }
}
} // namespace mlir::tc

namespace {
void SelectionNode::dump(int indent) {
  if (childs.empty()) {
    llvm::errs() << "\n";
    return;
  }
  if (axesSelections.empty()) {
    for (auto [idx, child] : llvm::enumerate(childs)) {
      if (idx)
        llvm::errs().indent(indent);
      llvm::errs() << "*";
      child->dump(indent + 1);
    }
  } else {
    for (auto [idx, child] : llvm::enumerate(childs)) {
      if (idx)
        llvm::errs().indent(indent);
      auto &selection = axesSelections[idx];
      std::string toPrint = "-> ";
      if (selection.size() != 1) {
        toPrint += "(";
      }
      for (auto [idx2, axis] : llvm::enumerate(selection)) {
        if (idx2 != 0) {
          toPrint += ", ";
        }
        toPrint += std::to_string(axis);
      }
      if (selection.size() != 1) {
        toPrint += ")";
      }
      toPrint += " ";
      llvm::errs() << toPrint;
      child->dump(indent + toPrint.size());
    }
  }
}
void SelectionNode::dumpOrder(int64_t orderID) {
  if (childs.empty())
    return;
  if (axesSelections.empty()) {
    llvm::errs() << "[";
    for (auto [idx, child] : llvm::enumerate(childs)) {
      if (idx)
        llvm::errs() << ", ";
      auto numChains = child->getNumberOfChains();
      auto childID = orderID % numChains;
      orderID = orderID / numChains;
      llvm::errs() << "(";
      child->dumpOrder(childID);
      llvm::errs() << ")";
    }
    llvm::errs() << "]";
  } else {
    for (auto [idx, child] : llvm::enumerate(childs)) {
      auto numChains = child->getNumberOfChains();
      if (orderID < numChains) {
        auto &selection = axesSelections[idx];
        if (selection.size() == 1)
          llvm::errs() << selection[0];
        else {
          llvm::errs() << "{";
          llvm::interleaveComma(selection, llvm::errs());
          llvm::errs() << "}";
        }
        llvm::errs() << ", ";
        child->dumpOrder(orderID);
        break;
      }
      orderID -= numChains;
    }
  }
}
} // namespace
