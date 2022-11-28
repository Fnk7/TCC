#ifndef TENSORCONTRACTION_CONTRACTIONTREE_H
#define TENSORCONTRACTION_CONTRACTIONTREE_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

using llvm::DenseMap;
using llvm::SmallVector;
using llvm::StringRef;

namespace contraction {
class Node;
class ContractionTree;

using IndexTy = int64_t;
using SizesTy = SmallVector<int, 4>;
using IndexesTy = SmallVector<int64_t, 4>;
using NodePtr = std::unique_ptr<Node>;
using ChildNodesTy = SmallVector<NodePtr, 2>;

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
  enum NodeKind {
    NK_Root,
    NK_Contraction,
    NK_Arith,
    NK_Addition,
    NK_Multiplication,
    NK_Leaf,
    NK_TreeAsLeaf,
    NK_InputAsLeaf
  };

private:
  const NodeKind Kind;

protected:
  SizesTy Sizes;
  IndexesTy Indexes;
  std::weak_ptr<InputTensor> InputLeaf;
  std::weak_ptr<ContractionTree> TreeLeaf;
  ChildNodesTy Childs;

  Node(NodeKind Kind, NodePtr Child) : Kind(Kind) {
    Childs.emplace_back(std::move(Child));
  }
  Node(NodeKind Kind, ChildNodesTy &Childs) : Kind(Kind) {
    std::move(Childs.begin(), Childs.end(), std::back_inserter(this->Childs));
  }
  Node(IndexesTy &Indexes, std::weak_ptr<ContractionTree> TreeLeaf)
      : Kind(NK_TreeAsLeaf), Indexes(Indexes), TreeLeaf(TreeLeaf) {}
  Node(IndexesTy &Indexes, std::weak_ptr<InputTensor> InputLeaf)
      : Kind(NK_InputAsLeaf), Indexes(Indexes), InputLeaf(InputLeaf) {}

  void dump(int indent);
  void dumpChilds(int indent);

public:
  NodeKind getKind() const { return Kind; }
  Node(NodeKind K) : Kind(K) {}

  virtual ~Node() = default;

public:
  template <typename... Args>
  static auto createNode(NodeKind Kind, Args &&...args) {
    return std::unique_ptr<Node>(new Node(Kind, std::forward<Args>(args)...));
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
  Node &getSingleChild() { return *(Childs[0]); }
  std::shared_ptr<InputTensor> getInputTensor() { return InputLeaf.lock(); }
  std::shared_ptr<ContractionTree> getContractionTensor() {
    return TreeLeaf.lock();
  }

  void dump() { dump(0); }
};

class RootNode : public Node {
private:
  SizesTy Sizes;
  IndexesTy IndexDefs;
  RootNode(SizesTy &Sizes, IndexesTy &Indexes, NodePtr Child)
      : Node(NK_Root, std::move(Child)), Sizes(Sizes), IndexDefs(Indexes) {}

public:
  static NodePtr get(SizesTy &Sizes, IndexesTy &Indexes, NodePtr Child) {
    return std::unique_ptr<Node>(
        new RootNode(Sizes, Indexes, std::move(Child)));
  }
  // Functions
  Node &getChild() { return *Childs[0]; }
  SizesTy &getSizes() { return Sizes; }
  IndexesTy &getIndexes() { return IndexDefs; }
  // For LLVM RTTI
  static bool classof(const Node *N) { return N->getKind() == NK_Root; }
  // dump
  void dump(int);
};

class ContractionNode : public Node {
private:
  SizesTy Sizes;
  IndexesTy IndexDefs;
  ContractionNode(SizesTy &Sizes, IndexesTy &Indexes, NodePtr Child)
      : Node(NK_Contraction, std::move(Child)), Sizes(Sizes),
        IndexDefs(Indexes) {}

public:
  static NodePtr get(SizesTy &Sizes, IndexesTy &Indexes, NodePtr Child) {
    return std::unique_ptr<Node>(
        new ContractionNode(Sizes, Indexes, std::move(Child)));
  }
  // Funtions
  Node &getChild() { return *Childs[0]; }
  SizesTy &getSizes() { return Sizes; }
  IndexesTy &getIndexes() { return IndexDefs; }
  // For LLVM RTTI
  static bool classof(const Node *N) { return N->getKind() == NK_Contraction; }
  // dump
  void dump(int);
};

class ArithNode : public Node {
protected:
  ArithNode(NodeKind Kind, ChildNodesTy &Childs) : Node(Kind, Childs) {}

public:
  static bool classof(const Node *N) {
    return N->getKind() == NK_Addition || N->getKind() == NK_Multiplication;
  }
};

class AdditionNode : public ArithNode {
private:
  AdditionNode(ChildNodesTy &Childs) : ArithNode(NK_Addition, Childs) {}

public:
  static NodePtr get(ChildNodesTy &Childs) {
    return std::unique_ptr<Node>(new AdditionNode(Childs));
  }
  // For LLVM RTTI
  static bool classof(const Node *N) { return N->getKind() == NK_Addition; }
};

class MultiplicationNode : public ArithNode {
private:
  MultiplicationNode(ChildNodesTy &Childs)
      : ArithNode(NK_Multiplication, Childs) {}

public:
  static NodePtr get(ChildNodesTy &Childs) {
    return std::unique_ptr<Node>(new MultiplicationNode(Childs));
  }
  // For LLVM RTTI
  static bool classof(const Node *N) { return N->getKind() == NK_Multiplication; }
};

class LeafNode : public Node {
protected:
  IndexesTy IndexUses;
  LeafNode(NodeKind Kind, IndexesTy &IndexUses)
      : Node(Kind), IndexUses(IndexUses) {}

public:
  // Member Functions
  IndexesTy &getIndexUses() { return IndexUses; }
  // For LLVM RTTI
  static bool classof(const Node *N) {
    return N->getKind() == NK_TreeAsLeaf || N->getKind() == NK_InputAsLeaf;
  }
};

using TreePtrTy = std::shared_ptr<ContractionTree>;
using TreeRefTy = std::weak_ptr<ContractionTree>;
class TreeAsLeafNode : public LeafNode {
private:
  TreeRefTy TreeRef;
  TreeAsLeafNode(IndexesTy &IndexUses, TreeRefTy TreeRef)
      : LeafNode(NK_TreeAsLeaf, IndexUses), TreeRef(TreeRef) {}

public:
  static NodePtr get(IndexesTy &IndexUses, TreeRefTy TreeRef) {
    return std::unique_ptr<Node>(new TreeAsLeafNode(IndexUses, TreeRef));
  }
  // Member Functions
  TreePtrTy getTree() { return TreeRef.lock(); }
  // For LLVM RTTI
  static bool classof(const Node *N) { return N->getKind() == NK_TreeAsLeaf; }
  // dump
  void dump(int);
};

using InputPtrTy = std::shared_ptr<InputTensor>;
using InputRefTy = std::weak_ptr<InputTensor>;
class InputAsLeafNode : public LeafNode {
private:
  InputRefTy InputRef;
  InputAsLeafNode(IndexesTy &IndexUses, InputRefTy InputRef)
      : LeafNode(NK_InputAsLeaf, IndexUses), InputRef(InputRef) {}

public:
  static NodePtr get(IndexesTy &IndexUses, InputRefTy InputRef) {
    return std::unique_ptr<Node>(new InputAsLeafNode(IndexUses, InputRef));
  }
  // Member Functions
  InputPtrTy getInput() { return InputRef.lock(); }
  // For LLVM RTTI
  static bool classof(const Node *N) { return N->getKind() == NK_InputAsLeaf; }
  // dump
  void dump(int);
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
  RootNode &getRootNode() { return llvm::cast<RootNode>(*Root); }
  StringRef getName() { return Name; }
  StringRef getUniqueName() { return UniqueName; }
  std::unique_ptr<Node> &getRoot() { return Root; }
  SizesTy &getSizes() { return getRootNode().getSizes(); }
  int getDimension() { return getSizes().size(); }
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
  SizesTy InputSizeMap;
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
    int InSz = 0;
    for (size_t i = 0; i < Sizes.size(); i++) {
      if ((int)i == Sizes[i]) {
        InputSizes.emplace_back(i);
        InputSizeMap.emplace_back(InSz++);
      } else {
        int UniqueSz = Sizes[i];
        Sizes[i] = Sizes[UniqueSz];
        InputSizeMap.emplace_back(InputSizeMap[Sizes[i]]);
      }
    }
  }
  int getInputSizeOf(int Sz) { return InputSizeMap[Sz]; }
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