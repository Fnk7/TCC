#ifndef TENSORCONTRACTION_CONTRACTIONTREE_H
#define TENSORCONTRACTION_CONTRACTIONTREE_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

using llvm::DenseMap;
using llvm::SmallVector;
using llvm::StringRef;

namespace contraction {
class Node;
class ContractionTree;

using SizesTy = SmallVector<int, 4>;
using IndexesTy = SmallVector<int, 4>;
using ChildNodesTy = SmallVector<std::unique_ptr<Node>, 2>;

class InputTensor {
  std::string Name;
  SizesTy Sizes;

public:
  InputTensor(std::string Name, SizesTy Sizes) : Name(Name), Sizes(Sizes) {}
  StringRef getName() { return Name; }
  SizesTy &getSizes() { return Sizes; }
  int getDimension() { return Sizes.size(); }
  void dump();
};

class Node {
public:
  enum NodeType : int {
    Root = 0,
    Contraction = 1,
    Addition = 2,
    Multiplication = 3,
    TreeAsLeaf = 4,
    InputAsLeaf = 5,
  };
  NodeType Type;

private:
  SizesTy Sizes;
  IndexesTy Indexes;
  std::weak_ptr<InputTensor> InputLeaf;
  std::weak_ptr<ContractionTree> TreeLeaf;
  ChildNodesTy Childs;

  Node(NodeType Type, SizesTy &Sizes, IndexesTy &Indexes,
       std::unique_ptr<Node> Child)
      : Type(Type), Sizes(Sizes), Indexes(Indexes) {
    Childs.emplace_back(std::move(Child));
  }
  Node(NodeType Type, ChildNodesTy &Childs) : Type(Type) {
    std::move(Childs.begin(), Childs.end(), std::back_inserter(this->Childs));
  }
  Node(IndexesTy &Indexes, std::weak_ptr<ContractionTree> TreeLeaf)
      : Type(TreeAsLeaf), Indexes(Indexes), TreeLeaf(TreeLeaf) {}
  Node(IndexesTy &Indexes, std::weak_ptr<InputTensor> InputLeaf)
      : Type(InputAsLeaf), Indexes(Indexes), InputLeaf(InputLeaf) {}

  void dump(int indent);
  void dumpChilds(int indent);

public:
  template <typename... Args>
  static auto createNode(NodeType Type, Args &&...args) {
    return std::unique_ptr<Node>(new Node(Type, std::forward<Args>(args)...));
  }

  template <typename... Args> static auto createRootNode(Args &&...args) {
    return createNode(Root, std::forward<Args>(args)...);
  }
  template <typename... Args>
  static auto createContractionNode(Args &&...args) {
    return createNode(Contraction, std::forward<Args>(args)...);
  }
  template <typename... Args> static auto createAdditionNode(Args &&...args) {
    return createNode(Addition, std::forward<Args>(args)...);
  }
  template <typename... Args>
  static auto createMultiplicationNode(Args &&...args) {
    return createNode(Multiplication, std::forward<Args>(args)...);
  }
  template <typename... Args> static auto createLeafNode(Args &&...args) {
    return std::unique_ptr<Node>(new Node(std::forward<Args>(args)...));
  }

  void setContractionTree(std::shared_ptr<ContractionTree> tree) {
    TreeLeaf = tree;
  }

  SizesTy &getSizes() { return Sizes; }
  IndexesTy &getIndexes() { return Indexes; }
  ChildNodesTy &getChilds() { return Childs; }
  std::shared_ptr<InputTensor> getInputTensor() { return InputLeaf.lock(); }
  std::shared_ptr<ContractionTree> getContractionTensor() {
    return TreeLeaf.lock();
  }

  void dump() { dump(0); };
};

class ContractionTree {
  std::string Name;
  std::string UniqueName;
  std::unique_ptr<Node> Root;

  ContractionTree(StringRef Name, std::unique_ptr<Node> Root)
      : Name(std::string(Name)), UniqueName(std::string(Name)),
        Root(std::move(Root)) {}

public:
  static std::shared_ptr<ContractionTree> create(StringRef Name,
                                                 std::unique_ptr<Node> Root) {
    auto CT = std::shared_ptr<ContractionTree>(
        new ContractionTree(Name, std::move(Root)));
    CT->Root->setContractionTree(CT);
    return CT;
  }
  StringRef getName() { return Name; }
  StringRef getUniqueName() { return UniqueName; }
  std::unique_ptr<Node> &getRoot() { return Root; }
  SizesTy &getSizes() { return Root->getSizes(); }
  int getDimension() { return Root->getSizes().size(); }
  void setUniqueName(StringRef UniqueName) {
    this->UniqueName = std::string(UniqueName);
  }
  void dump() { Root->dump(); };
};

using InputTensorsTy = SmallVector<std::shared_ptr<InputTensor>>;
using InputTensorMapTy = DenseMap<StringRef, std::shared_ptr<InputTensor>>;
using ContractionTreesTy = SmallVector<std::shared_ptr<ContractionTree>>;
using ContractionTreeMapTy =
    DenseMap<StringRef, std::shared_ptr<ContractionTree>>;

class ContractionForest {
  std::string Name;
  // Inputs and outputs and reserves
  SizesTy InputSizes;
  InputTensorsTy InputTensors;
  InputTensorsTy ReservedInputs;
  ContractionTreesTy OutputTrees;
  // Shapes and trees
  SizesTy Sizes;
  InputTensorMapTy InputMap;
  ContractionTreesTy Trees;
  ContractionTreeMapTy TreeMap;

public:
  ContractionForest() { Name = "Contraction"; }
  void dump();

  std::shared_ptr<InputTensor> getInputTensor(StringRef Name) {
    if (InputMap.count(Name)) {
      return InputMap[Name];
    }
    return nullptr;
  }
  std::shared_ptr<InputTensor> getOrInsertInputTensor(StringRef Name, int dim) {
    std::shared_ptr<InputTensor> IT;
    if (InputMap.count(Name)) {
      IT = InputMap[Name];
      assert(IT->getDimension() == dim);
    } else {
      SizesTy InputSizes;
      for (int Sz = 0; Sz < dim; Sz++)
        InputSizes.emplace_back(Sizes.size() + Sz);
      Sizes.append(InputSizes);
      IT = std::make_shared<InputTensor>(std::string(Name), InputSizes);
      InputTensors.emplace_back(IT);
      InputMap[IT->getName()] = IT;
    }
    return IT;
  }

  std::shared_ptr<ContractionTree> getContractionTree(StringRef Name) {
    if (TreeMap.count(Name)) {
      return TreeMap[Name];
    }
    return nullptr;
  }
  void addContractionTree(std::shared_ptr<ContractionTree> CT) {
    Trees.emplace_back(CT);
    int Suffix = 0;
    while (TreeMap.count(CT->getUniqueName())) {
      CT->setUniqueName(std::string(CT->getName()) + '#' +
                        std::to_string(++Suffix));
    }
    TreeMap[CT->getUniqueName()] = CT;
    TreeMap[CT->getName()] = CT;
  }

  void addKeep(StringRef Name) {
    if (auto CT = getContractionTree(Name)) {
      OutputTrees.emplace_back(CT);
    } else if (auto IT = getInputTensor(Name)) {
      ReservedInputs.emplace_back(IT);
    } else {
      assert("Input or Tree Tensor not defined!");
    }
  }
  // Sizes
  inline void setSizeEqual(int &Sz1, int &Sz2) {
    assert(Sz1 < (int)Sizes.size());
    assert(Sz2 < (int)Sizes.size() && Sz2 >= 0);
    if (Sz1 >= 0 && Sz1 != Sz2) {
      if (Sizes[Sz1] < Sizes[Sz2]) {
        Sizes[Sz2] = Sizes[Sz1];
      } else {
        Sizes[Sz1] = Sizes[Sz2];
      }
    }
    Sz1 = Sizes[Sz2];
  }
  void setSizeEqual(SizesTy &Sz1, SizesTy &Sz2) {
    assert(Sz1.size() == Sz2.size());
    for (size_t i = 0; i < Sz1.size(); i++) {
      setSizeEqual(Sz1[i], Sz2[i]);
    }
  }
  void resolveInputSizes() {
    for (size_t i = 0; i < Sizes.size(); i++) {
      if ((int)i == Sizes[i]) {
        InputSizes.emplace_back(i);
      } else {
        int UniqueSz = Sizes[i];
        Sizes[i] = Sizes[UniqueSz];
      }
    }
  }
  int getUniqueInputSizeOf(int Sz) { return Sizes[Sz]; }

  StringRef getName() { return Name; }
  SizesTy &getInputSizes() { return InputSizes; }
  InputTensorsTy &getInputTensors() { return InputTensors; }
  ContractionTreesTy &getOutputTensors() { return OutputTrees; }
  ContractionTreesTy &getContractionTrees() { return Trees; }

  // TODO:

  int getNumInputSizs() {
    int num = 0;
    for (size_t i = 0; i < Sizes.size(); i++)
      if ((int)i == Sizes[i])
        num++;
    return num;
  }

  int getNumInputTensors() { return InputMap.size(); }
  int getNumOutputTensors() { return OutputTrees.size(); }
};
} // namespace contraction

#endif