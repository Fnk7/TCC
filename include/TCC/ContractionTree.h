#ifndef TENSORCONTRACTION_CONTRACTIONTREE_H
#define TENSORCONTRACTION_CONTRACTIONTREE_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

using llvm::raw_ostream;

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
using IndexesTy = SmallVector<int64_t, 4>;
using ChildNodesTy = SmallVector<NodePtr, 2>;

class SizeSymbol;
using SizeID = int64_t;
using SizePtr = SizeSymbol *;
// SizeID map to SizeSymbol
using SizesMapTy = SmallVector<std::unique_ptr<SizeSymbol>>;
class SizeSymbol : public Symbol {
private:
  std::string Name;
  int64_t UpperBound;
  int64_t LowerBound;
  SizeID ID;
  SmallVector<SizePtr> Aliases;
  bool Implicit;

public:
  SizeSymbol(SizeID ID) : Symbol(SK_Size) {
    Aliases.emplace_back(this);
    this->ID = ID;
    Name = "Sz." + std::to_string(ID);
    Implicit = true;
    UpperBound = 0;
    LowerBound = 0;
  }
  SizeSymbol(SizeID ID, StringRef Name, int64_t UB = 0, int64_t LB = 0)
      : Symbol(SK_Size), Name(Name) {
    Aliases.emplace_back(this);
    this->ID = ID;
    Implicit = false;
    UpperBound = UB;
    LowerBound = LB;
  }
  // Member Functions
  bool operator==(const SizeSymbol &Sz) { return this == &Sz; }
  static bool makeSizeEqual(SizeSymbol &Sz1, SizeSymbol &Sz2) {
    if (Sz1 == Sz2)
      return true;
    if (Sz1.Implicit && !Sz2.Implicit) {
      return makeSizeEqual(Sz2, Sz1);
    } else {
      assert(&Sz1 == Sz1.Aliases[0]);
      assert(&Sz2 == Sz2.Aliases[0]);
      size_t NewAlias = Sz1.Aliases.size() + 1;
      Sz1.Aliases.append(Sz2.Aliases);
      Sz2.Aliases.clear();
      Sz2.Aliases.emplace_back(&Sz1);
      for (; NewAlias < Sz1.Aliases.size(); NewAlias++) {
        Sz1.Aliases[NewAlias]->Aliases[0] = &Sz1;
      }
      return true;
    }
  }
  void setUpperBound(int64_t UB) { UpperBound = UB; }
  void setLowerBound(int64_t LB) { LowerBound = LB; }
  SizeSymbol &get() { return *Aliases[0]; }
  bool isUnique() { return this == &get(); }
  SizeID getUID() { return get().ID; }
  int64_t getUpperBound() { return get().UpperBound; }
  int64_t getLowerBound() { return get().LowerBound; }
  // For LLVM RTTI
  static bool classof(const Symbol *S) { return S->getKind() == SK_Size; }
  // Print
  void dump();
};

using SizesTy = SmallVector<SizePtr, 4>;

class InputTensor : public Symbol {
private:
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
  friend raw_ostream &operator<<(raw_ostream &OS, NodeKind K) {
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
  CTreePtrsTy OutputCTrees;
  // Shapes and trees
  CTreesTy Trees;
  // Symbols
  SymbolsMapTy SymbolsMap;
  // Input Tensors
  InTensorsMapTy InTensorsMap;
  // Contraction Tree Tensors
  CTreesMapTy CTreesMap;
  // Sizes
  SizesMapTy SizesMap;

public:
  ContractionForest() { Name = "CT"; }
  void dump();
  // Get Tensor
  SymbolPtr getTensorOfName(StringRef Name) {
    return SymbolsMap.count(Name) ? SymbolsMap[Name] : nullptr;
  }
  InTensorPtr getInputTensor(StringRef Name) {
    return InTensorsMap.count(Name) ? InTensorsMap[Name] : nullptr;
  }
  CTreePtr getContractionTree(StringRef Name) {
    return CTreesMap.count(Name) ? CTreesMap[Name] : nullptr;
  }
  // Add Tensor
  InTensorPtr addInputTensor(StringRef Name, int Dim) {
    assert(!InTensorsMap.count(Name));
    SizesTy InputSizes;
    SizeID ID = SizesMap.size();
    for (int Sz = 0; Sz < Dim; Sz++) {
      SizesMap.emplace_back(new SizeSymbol(ID));
      InputSizes.emplace_back(SizesMap.back().get());
    }
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
  // Add Keep
  void addKeep(StringRef Name) {
    if (auto CT = getContractionTree(Name)) {
      OutputCTrees.emplace_back(CT);
    } else if (auto IT = getInputTensor(Name)) {
      ReservedInputs.emplace_back(IT);
    } else {
      llvm_unreachable("Input or CTree Tensor not defined!");
    }
  }
  // Sizes
  void setSizeEqual(SizesTy &Sz1, SizesTy &Sz2) {
    assert(Sz1.size() == Sz2.size());
    for (size_t i = 0; i < Sz1.size(); i++) {
      if (!Sz1[i]) {
        assert(Sz2[i]);
        Sz1[i] = &(Sz2[i]->get());
      } else if (!Sz2[i]) {
        Sz2[i] = &(Sz1[i]->get());
      } else {
        SizeSymbol::makeSizeEqual(Sz1[i]->get(), Sz2[i]->get());
      }
    }
  }
  void resolveInputSizes() {
    // TODO
  }
  int64_t getInputSizeOf(SizePtr Sz) { return Sz->getUID(); }

  StringRef getName() { return Name; }
  SizesTy &getInputSizes() { return InputSizes; }
  InTensorsTy &getInputTensors() { return InputTensors; }
  CTreePtrsTy &getOutputTensors() { return OutputCTrees; }
  CTreesTy &getContractionTrees() { return Trees; }

  // TODO:

  int getNumInputSizs() {
    int num = 0;
    for (auto &Sz : SizesMap) {
      if (Sz->isUnique())
        num++;
    }
    return num;
  }

  int getNumInputTensors() { return InputTensors.size(); }
  int getNumOutputTensors() { return OutputCTrees.size(); }
};
} // namespace contraction

#endif