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
// Node - AST node of Contraction Expression. It use the LLVM style RTTI.
// And since LLVM Value class use deleteValue and ValueDeleter for unique_ptr,
// std default_delete<Node> was not used here.
class Node;
struct NodeDeleter {
  void operator()(Node *N);
};
using NodePtr = std::unique_ptr<Node, NodeDeleter>;

// Symbol - Represent the indentity in Contraction Expression and its kind,
// include input tensor, contraction tensor, index, constant index and size.
class Symbol;
using SymbolPtr = Symbol *;
using SymbolsMapTy = DenseMap<StringRef, SymbolPtr>;

// InputTensor - The arrays used by Contraction Expression that not
// defined by predecessor expressions.
class InputTensor;
using InTensorsTy = SmallVector<std::unique_ptr<InputTensor>>;
using InTensorPtr = InputTensor *;
using InTensorPtrsTy = SmallVector<InTensorPtr>;
using InTensorsMapTy = DenseMap<StringRef, InTensorPtr>;

// ContractionTree - Line of the Contraction Expression, which do
// some calculation (contraction is major) and store value to a tensor. The
// result can be used by succeed Expression.
class ContractionTree;
using CTreesTy = SmallVector<std::unique_ptr<ContractionTree>>;
using CTreePtr = ContractionTree *;
using CTreePtrsTy = SmallVector<CTreePtr>;
using CTreesMapTy = DenseMap<StringRef, CTreePtr>;

class Symbol {
public:
  enum SymbolKind { SK_CTensor, SK_InTenser, SK_Index, SK_ConstIndex, SK_Size };

protected:
  SymbolKind Kind;
  Symbol(SymbolKind K) : Kind(K) {}

public:
  SymbolKind getKind() const { return Kind; }
};

using IndexTy = int64_t;
using SizesTy = SmallVector<int, 4>;
using IndexesTy = SmallVector<int64_t, 4>;
using ChildNodesTy = SmallVector<NodePtr, 2>;

class InputTensor : public Symbol {
  std::string Name;
  SizesTy Sizes;

public:
  InputTensor(StringRef Name, SizesTy Sizes)
      : Symbol(SK_InTenser), Name(Name), Sizes(Sizes) {}
  StringRef getName() { return Name; }
  SizesTy &getSizes() { return Sizes; }
  int getDimension() { return Sizes.size(); }
  // For LLVM RTTI
  static bool classof(const Symbol *S) { return S->getKind() == SK_InTenser; }
  // dump
  void dump();
};

class Node {
public:
  // NodeKind and LLVM RTTI
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

protected:
  const NodeKind Kind;
  ChildNodesTy Childs;
  Node(NodeKind K) : Kind(K) {}
  Node(NodeKind Kind, NodePtr Child) : Node(Kind) {
    Childs.emplace_back(std::move(Child));
  }
  Node(NodeKind Kind, ChildNodesTy &Childs) : Node(Kind) {
    std::move(Childs.begin(), Childs.end(), std::back_inserter(this->Childs));
  }
  ~Node() = default;

public:
  NodeKind getKind() const { return Kind; }
  ChildNodesTy &getChilds() { return Childs; }
  void deleteNode();
  // Print
  void dump() { dump(0); }
  void dump(int indent);
  void dumpChilds(int indent);
  static const std::string getKindString(NodeKind K);
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, NodeKind K) {
    OS << getKindString(K);
    return OS;
  }
};

// RootNode: Write result into tensor, and define indexes.
class RootNode : public Node {
private:
  SizesTy Sizes;
  IndexesTy IndexDefs;
  RootNode(SizesTy &Sizes, IndexesTy &Indexes, NodePtr Child)
      : Node(NK_Root, std::move(Child)), Sizes(Sizes), IndexDefs(Indexes) {}

public:
  static NodePtr get(SizesTy &Sizes, IndexesTy &Indexes, NodePtr Child) {
    return NodePtr(new RootNode(Sizes, Indexes, std::move(Child)));
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

// ContractionNode: Contract tensor in some axes, where the indexes defined for.
class ContractionNode : public Node {
private:
  SizesTy Sizes;
  IndexesTy IndexDefs;
  ContractionNode(SizesTy &Sizes, IndexesTy &Indexes, NodePtr Child)
      : Node(NK_Contraction, std::move(Child)), Sizes(Sizes),
        IndexDefs(Indexes) {}

public:
  static NodePtr get(SizesTy &Sizes, IndexesTy &Indexes, NodePtr Child) {
    return NodePtr(new ContractionNode(Sizes, Indexes, std::move(Child)));
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

// ArithNode: addition, negative or multiplication.
class ArithNode : public Node {
protected:
  ArithNode(NodeKind Kind, ChildNodesTy &Childs) : Node(Kind, Childs) {}

public:
  static bool classof(const Node *N) {
    return N->getKind() == NK_Addition || N->getKind() == NK_Multiplication;
  }
};

// AdditionNode: tensor addition, more than elementwise.
class AdditionNode : public ArithNode {
private:
  AdditionNode(ChildNodesTy &Childs) : ArithNode(NK_Addition, Childs) {}

public:
  static NodePtr get(ChildNodesTy &Childs) {
    return NodePtr(new AdditionNode(Childs));
  }
  // For LLVM RTTI
  static bool classof(const Node *N) { return N->getKind() == NK_Addition; }
};

// AdditionNode: tensor multiplication, more than elementwise.
class MultiplicationNode : public ArithNode {
private:
  MultiplicationNode(ChildNodesTy &Childs)
      : ArithNode(NK_Multiplication, Childs) {}

public:
  static NodePtr get(ChildNodesTy &Childs) {
    return NodePtr(new MultiplicationNode(Childs));
  }
  // For LLVM RTTI
  static bool classof(const Node *N) {
    return N->getKind() == NK_Multiplication;
  }
};

// LeafNode: Reference of a tensor either input or contraction expression
// result.
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

// TreeAsLeafNode: Reference to a contraction expression result.
class TreeAsLeafNode : public LeafNode {
private:
  CTreePtr TreeRef;
  TreeAsLeafNode(IndexesTy &IndexUses, CTreePtr TreeRef)
      : LeafNode(NK_TreeAsLeaf, IndexUses), TreeRef(TreeRef) {}

public:
  static NodePtr get(IndexesTy &IndexUses, CTreePtr TreeRef) {
    return NodePtr(new TreeAsLeafNode(IndexUses, TreeRef));
  }
  // Member Functions
  CTreePtr getTree() { return TreeRef; }
  // For LLVM RTTI
  static bool classof(const Node *N) { return N->getKind() == NK_TreeAsLeaf; }
  // dump
  void dump(int);
};

// InputAsLeafNode: Reference to an input tensor.
class InputAsLeafNode : public LeafNode {
private:
  InTensorPtr InputRef;
  InputAsLeafNode(IndexesTy &IndexUses, InTensorPtr InputRef)
      : LeafNode(NK_InputAsLeaf, IndexUses), InputRef(InputRef) {}

public:
  static NodePtr get(IndexesTy &IndexUses, InTensorPtr InputRef) {
    return NodePtr(new InputAsLeafNode(IndexUses, InputRef));
  }
  // Member Functions
  InTensorPtr getInput() { return InputRef; }
  // For LLVM RTTI
  static bool classof(const Node *N) { return N->getKind() == NK_InputAsLeaf; }
  // dump
  void dump(int);
};

// LLVM RTTI unique_ptr release.
inline void NodeDeleter::operator()(Node *N) { N->deleteNode(); }
inline void Node::deleteNode() {
  // llvm::errs() << "Delete Node " << getKind() << "\n";
  if (auto *N = llvm::dyn_cast<RootNode>(this)) {
    delete N;
  } else if (auto *N = llvm::dyn_cast<ContractionNode>(this)) {
    delete N;
  } else if (auto *N = llvm::dyn_cast<AdditionNode>(this)) {
    delete N;
  } else if (auto *N = llvm::dyn_cast<MultiplicationNode>(this)) {
    delete N;
  } else if (auto *N = llvm::dyn_cast<TreeAsLeafNode>(this)) {
    delete N;
  } else if (auto *N = llvm::dyn_cast<InputAsLeafNode>(this)) {
    delete N;
  } else {
    llvm_unreachable("Deleting Undifined Node");
  }
}

class ContractionTree : public Symbol {
private:
  std::string Name;
  NodePtr Root;

public:
  ContractionTree(StringRef Name, NodePtr Root)
      : Symbol(SK_CTensor), Name(std::string(Name)), Root(std::move(Root)) {
    assert(llvm::isa<RootNode>(this->Root.get()));
  }
  RootNode &getRootNode() { return llvm::cast<RootNode>(*Root); }
  StringRef getName() { return Name; }
  SizesTy &getSizes() { return getRootNode().getSizes(); }
  int getDimension() { return getSizes().size(); }
  // For LLVM RTTI
  static bool classof(const Symbol *S) { return S->getKind() == SK_CTensor; }
  // dump
  void dump();
};

class ContractionForest {
  std::string Name;
  // Inputs and outputs and reserves
  SizesTy InputSizes;
  SizesTy InputSizeMap;
  InTensorsTy InputTensors;
  InTensorPtrsTy ReservedInputs;
  CTreePtrsTy OutputTrees;
  // Shapes and trees
  SizesTy Sizes;
  CTreesTy Trees;
  // Symbols
  SymbolsMapTy SymbolsMap;
  // Input Tensors
  InTensorsMapTy InTensorsMap;
  // Contraction Tree Tensors
  CTreesMapTy CTreesMap;

public:
  ContractionForest() { Name = "CT"; }
  void dump();

  SymbolPtr getTensorOfName(StringRef Name) {
    return SymbolsMap.count(Name) ? SymbolsMap[Name] : nullptr;
  }
  InTensorPtr getInputTensor(StringRef Name) {
    return InTensorsMap.count(Name) ? InTensorsMap[Name] : nullptr;
  }
  CTreePtr getContractionTree(StringRef Name) {
    return CTreesMap.count(Name) ? CTreesMap[Name] : nullptr;
  }

  InTensorPtr getOrInsertInputTensor(StringRef Name, int Dim) {
    InTensorPtr IT;
    if (InTensorsMap.count(Name)) {
      IT = InTensorsMap[Name];
      assert(IT->getDimension() == Dim);
    } else {
      SizesTy InputSizes;
      for (int Sz = 0; Sz < Dim; Sz++)
        InputSizes.emplace_back(Sizes.size() + Sz);
      Sizes.append(InputSizes);
      InputTensors.emplace_back(new InputTensor(Name, InputSizes));
      IT = InputTensors.back().get();
      SymbolsMap[IT->getName()] = IT;
      InTensorsMap[IT->getName()] = IT;
    }
    return IT;
  }
  InTensorPtr addInputTensor(StringRef Name, int Dim) {
    assert(!InTensorsMap.count(Name));
    SizesTy InputSizes;
    for (int Sz = 0; Sz < Dim; Sz++)
      InputSizes.emplace_back(Sizes.size() + Sz);
    Sizes.append(InputSizes);
    InputTensors.emplace_back(new InputTensor(Name, InputSizes));
    auto IT = InputTensors.back().get();
    SymbolsMap[IT->getName()] = IT;
    InTensorsMap[IT->getName()] = IT;
    return IT;
  }
  void addContractionTree(StringRef Name, NodePtr Root) {
    Trees.emplace_back(new ContractionTree(Name, std::move(Root)));
    auto CT = Trees.back().get();
    SymbolsMap[CT->getName()] = CT;
    CTreesMap[CT->getName()] = CT;
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
  InTensorsTy &getInputTensors() { return InputTensors; }
  CTreePtrsTy &getOutputTensors() { return OutputTrees; }
  CTreesTy &getContractionTrees() { return Trees; }

  // TODO:

  int getNumInputSizs() {
    int num = 0;
    for (size_t i = 0; i < Sizes.size(); i++)
      if ((int)i == Sizes[i])
        num++;
    return num;
  }

  int getNumInputTensors() { return InputTensors.size(); }
  int getNumOutputTensors() { return OutputTrees.size(); }
};
} // namespace contraction

#endif