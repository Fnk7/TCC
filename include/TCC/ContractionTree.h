#ifndef TENSORCONTRACTION_CONTRACTIONTREE_H
#define TENSORCONTRACTION_CONTRACTIONTREE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"

using llvm::raw_ostream;

using llvm::ArrayRef;
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
  enum SymbolKind {
    SK_CTensor,
    SK_InTenser,
    SK_IterIndex,
    SK_ParamIndex,
    SK_Size
  };

protected:
  const SymbolKind Kind;
  int UsedCount = 0;
  std::string Name;
  Symbol(SymbolKind K, StringRef Name) : Kind(K), Name(Name) {}

public:
  // Member Functions
  StringRef getName() { return Name; }
  // For LLVM RTTI
  SymbolKind getKind() const { return Kind; }
};

// using IndexTy = int64_t;
// using IndexesTy = SmallVector<int64_t, 4>;
using ChildNodesTy = SmallVector<NodePtr, 2>;

class SizeSymbol;
using SizeID = int64_t;
using SizePtr = SizeSymbol *;
using SizesTy = SmallVector<SizePtr, 4>;
using ShapeTy = SizesTy;
// SizeID map to SizeSymbol
using SizesMapTy = SmallVector<std::unique_ptr<SizeSymbol>>;
class SizeSymbol : public Symbol {
private:
  SizeID ID;
  bool Implicit;

  class EqualityStorage {
  public:
    SizeID ID;
    SizeID UID = -1;
    int64_t UpperBound = 0;
    int64_t LowerBound = 0;
    SizesTy Aliases;
    EqualityStorage(ArrayRef<SizePtr> Aliases) : Aliases(Aliases) {}
    void setID(SizeID ID) { this->ID = this->ID < ID ? this->ID : ID; }
  };
  EqualityStorage *EqualityInfo;

  EqualityStorage *getInfo() {
    if (!EqualityInfo) {
      EqualityInfo = new EqualityStorage{{this}};
      EqualityInfo->ID = ID;
    }
    return EqualityInfo;
  }

public:
  SizeSymbol(SizeID ID) : Symbol(SK_Size, ""), ID(ID) {
    Name = "Sz." + std::to_string(ID);
    Implicit = true;
    EqualityInfo = nullptr;
  }
  SizeSymbol(SizeID ID, StringRef Name, int64_t UB = 0, int64_t LB = 0)
      : Symbol(SK_Size, Name), ID(ID) {
    Implicit = false;
    EqualityInfo = nullptr;
    setUpperBound(UB);
    setLowerBound(LB);
  }
  // Member Functions
  bool operator==(const SizeSymbol &Sz) {
    return EqualityInfo && EqualityInfo == Sz.EqualityInfo;
  }
  static void makeSizeEqual(SizeSymbol &Sz1, SizeSymbol &Sz2) {
    if (Sz1 == Sz2)
      return;
    if (!Sz1.EqualityInfo && Sz2.EqualityInfo) {
      return makeSizeEqual(Sz2, Sz1);
    } else if (!Sz2.EqualityInfo) {
      Sz1.getInfo()->Aliases.emplace_back(&Sz2);
      Sz2.EqualityInfo = Sz1.EqualityInfo;
      Sz1.EqualityInfo->setID(Sz2.ID);
    } else {
      Sz1.EqualityInfo->setID(Sz2.EqualityInfo->ID);
      auto ToDelete = Sz2.EqualityInfo;
      for (auto Alias : Sz2.EqualityInfo->Aliases) {
        Sz1.EqualityInfo->Aliases.emplace_back(Alias);
        Alias->EqualityInfo = Sz1.EqualityInfo;
      }
      delete ToDelete;
    }
  }
  void setUpperBound(int64_t UB) { getInfo()->UpperBound = UB; }
  void setLowerBound(int64_t LB) { getInfo()->LowerBound = LB; }
  bool isFirstAppear() { return ID == getInfo()->ID; }
  void setUID(SizeID UID) { getInfo()->UID = UID; }
  SizeID getID() { return getInfo()->ID; }
  SizeID getUID() { return getInfo()->UID; }
  int64_t getUpperBound() { return getInfo()->UpperBound; }
  int64_t getLowerBound() { return getInfo()->LowerBound; }
  // For LLVM RTTI
  static bool classof(const Symbol *S) { return S->getKind() == SK_Size; }
  // Print
  void dump();
  // Destructor
  ~SizeSymbol() {
    if (EqualityInfo) {
      llvm::erase_value(EqualityInfo->Aliases, this);
      if (EqualityInfo->Aliases.empty())
        delete EqualityInfo;
    }
  }
};

class IndexSymbol;
class IterIndexSymbol;
using IndexID = int64_t;
using IndexTy = SymbolPtr;
using IndexPtr = IndexSymbol *;
using IndexesTy = SmallVector<IndexSymbol *>;
using IndexDefsTy = SmallVector<std::unique_ptr<IterIndexSymbol>>;
using IndexUsesTy = IndexesTy;
class IndexSymbol : public Symbol {
protected:
  SizePtr Size;
  IndexSymbol(SymbolKind K, StringRef Name) : Symbol(K, Name), Size(nullptr) {}

public:
  // Member Functions
  bool isImplicit() { return getName().substr(0, 3) == "ci#";}
  void setSize(SizePtr Size) { this->Size = Size; }
  SizePtr getSize() { return Size; }
  // For LLVM RTTI
  static bool classof(const Symbol *S) {
    return S->getKind() >= SK_IterIndex && S->getKind() <= SK_ParamIndex;
  }
  // Print
  void dump();
};

class IterIndexSymbol : public IndexSymbol {
private:
  IndexID ID;

public:
  IterIndexSymbol(IndexID ID, StringRef Name)
      : IndexSymbol(SK_IterIndex, Name), ID(ID) {}
  // Member Functions
  IndexID getID() { return ID; }
  // For LLVM RTTI
  static bool classof(const Symbol *S) { return S->getKind() == SK_IterIndex; }
  // Print
  void dump();
};

class ParamIndexSymbol;
using ParamIndexPtr = ParamIndexSymbol *;
class ParamIndexSymbol : public IndexSymbol {
private:
  int64_t Value;

public:
  static const int64_t Undefined = -1;
  ParamIndexSymbol(StringRef Name, int64_t Value)
      : IndexSymbol(SK_ParamIndex, Name), Value(Value) {}
  // Member Functions
  bool hasValue() { return Value >= 0; }
  int64_t getValue() {
    assert(hasValue());
    return Value;
  }
  // For LLVM RTTI
  static bool classof(const Symbol *S) { return S->getKind() == SK_ParamIndex; }
  // Print
  void dump();
};

class InputTensor : public Symbol {
private:
  ShapeTy Shape;

public:
  InputTensor(StringRef Name, ShapeTy Shape)
      : Symbol(SK_InTenser, Name), Shape(Shape) {}
  ShapeTy &getShape() { return Shape; }
  int64_t getDimension() { return Shape.size(); }
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
    NK_Negative,
    NK_Leaf,
    NK_Constant,
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
  ShapeTy Shape;
  IndexDefsTy IndexDefs;
  RootNode(IndexDefsTy &IndexDefs, NodePtr Child)
      : Node(NK_Root, std::move(Child)) {
    std::move(IndexDefs.begin(), IndexDefs.end(),
              std::back_inserter(this->IndexDefs));
    for (auto &IndexDef : this->IndexDefs) {
      Shape.emplace_back(IndexDef->getSize());
    }
  }

public:
  static NodePtr get(IndexDefsTy &IndexDefs, NodePtr Child) {
    return NodePtr(new RootNode(IndexDefs, std::move(Child)));
  }
  // Functions
  Node &getChild() { return *Childs[0]; }
  ShapeTy &getShape() { return Shape; }
  IndexDefsTy &getIndexes() { return IndexDefs; }
  // For LLVM RTTI
  static bool classof(const Node *N) { return N->getKind() == NK_Root; }
  // dump
  void dump(int);
};

// ContractionNode: Contract tensor in some axes, where the indexes defined for.
class ContractionNode : public Node {
private:
  SizesTy Sizes;
  IndexDefsTy IndexDefs;
  ContractionNode(IndexDefsTy &IndexDefs, NodePtr Child)
      : Node(NK_Contraction, std::move(Child)) {
    std::move(IndexDefs.begin(), IndexDefs.end(),
              std::back_inserter(this->IndexDefs));
  }

public:
  static NodePtr get(IndexDefsTy &IndexDefs, NodePtr Child) {
    return NodePtr(new ContractionNode(IndexDefs, std::move(Child)));
  }
  // Funtions
  Node &getChild() { return *Childs[0]; }
  SizesTy &getSizes() { return Sizes; }
  IndexDefsTy &getIndexes() { return IndexDefs; }
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
  SmallVector<bool> AddOrSub;
  AdditionNode(ChildNodesTy &Childs, SmallVector<bool> &AddOrSub)
      : ArithNode(NK_Addition, Childs), AddOrSub(AddOrSub) {}

public:
  static NodePtr get(ChildNodesTy &Childs, SmallVector<bool> &AddOrSub) {
    return NodePtr(new AdditionNode(Childs, AddOrSub));
  }
  // For LLVM RTTI
  static bool classof(const Node *N) { return N->getKind() == NK_Addition; }
  // dump
  void dump(int);
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
  IndexUsesTy IndexUses;
  LeafNode(NodeKind Kind, IndexUsesTy &IndexUses)
      : Node(Kind), IndexUses(IndexUses) {}

public:
  // Member Functions
  IndexUsesTy &getIndexUses() { return IndexUses; }
  IndexUsesTy getIterIndexUses() {
    IndexUsesTy IIndexUses;
    for (auto IndexUse : IndexUses) {
      if (llvm::isa<IterIndexSymbol>(IndexUse)) {
        IIndexUses.emplace_back(IndexUse);
      }
    }
    return IIndexUses;
  }
  // For LLVM RTTI
  static bool classof(const Node *N) {
    return N->getKind() == NK_TreeAsLeaf || N->getKind() == NK_InputAsLeaf;
  }
};

// TreeAsLeafNode: Reference to a contraction expression result.
class TreeAsLeafNode : public LeafNode {
private:
  CTreePtr TreeRef;
  TreeAsLeafNode(IndexUsesTy &IndexUses, CTreePtr TreeRef)
      : LeafNode(NK_TreeAsLeaf, IndexUses), TreeRef(TreeRef) {}

public:
  static NodePtr get(IndexUsesTy &IndexUses, CTreePtr TreeRef) {
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
  InputAsLeafNode(IndexUsesTy &IndexUses, InTensorPtr InputRef)
      : LeafNode(NK_InputAsLeaf, IndexUses), InputRef(InputRef) {}

public:
  static NodePtr get(IndexUsesTy &IndexUses, InTensorPtr InputRef) {
    return NodePtr(new InputAsLeafNode(IndexUses, InputRef));
  }
  // Member Functions
  InTensorPtr getInput() { return InputRef; }
  // For LLVM RTTI
  static bool classof(const Node *N) { return N->getKind() == NK_InputAsLeaf; }
  // dump
  void dump(int);
};

// NegativeNode:
class NegativeNode : public Node {
private:
  NegativeNode(NodePtr Child) : Node(NK_Negative, std::move(Child)) {}

public:
  static NodePtr get(NodePtr Child) {
    return NodePtr(new NegativeNode(std::move(Child)));
  }
  // For LLVM RTTI
  static bool classof(const Node *N) { return N->getKind() == NK_Negative; }
  // dump
  void dump(int);
};

// NegativeNode:
class ConstantNode : public Node {
private:
  double Value;
  ConstantNode(double Value) : Node(NK_Constant), Value(Value) {}

public:
  static NodePtr get(double Value) { return NodePtr(new ConstantNode(Value)); }
  // For LLVM RTTI
  static bool classof(const Node *N) { return N->getKind() == NK_Constant; }
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
  } else if (auto *N = llvm::dyn_cast<NegativeNode>(this)) {
    delete N;
  } else if (auto *N = llvm::dyn_cast<ConstantNode>(this)) {
    delete N;
  } else {
    llvm_unreachable("Deleting Undifined Node");
  }
}

class ContractionTree : public Symbol {
private:
  NodePtr Root;

public:
  ContractionTree(StringRef Name, NodePtr Root)
      : Symbol(SK_CTensor, Name), Root(std::move(Root)) {
    assert(llvm::isa<RootNode>(this->Root.get()));
  }
  RootNode &getRootNode() { return llvm::cast<RootNode>(*Root); }
  ShapeTy &getShape() { return getRootNode().getShape(); }
  int64_t getDimension() { return getShape().size(); }
  // For LLVM RTTI
  static bool classof(const Symbol *S) { return S->getKind() == SK_CTensor; }
  // dump
  void dump();
};

#define DEBUG_TYPE "TCForest"

using ParamIndicesHoldTy = SmallVector<std::unique_ptr<ParamIndexSymbol>>;
class ContractionForest {
  std::string Name;
  // Symbols
  SymbolsMapTy SymbolsMap;
  // Const Indexes
  // Input Tensors
  InTensorsTy InputTensors;
  InTensorsMapTy InTensorsMap;
  InTensorPtrsTy ReservedInputs;
  // Contraction Tree Tensors
  CTreesTy CTrees;
  CTreesMapTy CTreesMap;
  CTreePtrsTy OutputCTrees;
  // Sizes
  size_t NumInputSizes;
  SizesMapTy SizeIDSymbolMap;
  // For resource release
  ParamIndicesHoldTy ParamIndices;

public:
  ContractionForest() { Name = "CT"; }
  void dump();
  // Get Tensor
  SymbolPtr getSymbolOfName(StringRef Name) {
    return SymbolsMap.count(Name) ? SymbolsMap[Name] : nullptr;
  }
  InTensorPtr getInputTensor(StringRef Name) {
    return InTensorsMap.count(Name) ? InTensorsMap[Name] : nullptr;
  }
  CTreePtr getContractionTree(StringRef Name) {
    return CTreesMap.count(Name) ? CTreesMap[Name] : nullptr;
  }
  // Done 12-08
  void addParamIndexDefine(int64_t Value, StringRef Name = "") {
    if (!Name.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "Add param index: " << Name);
      LLVM_DEBUG(if (Value >= 0) llvm::dbgs() << " = " << Value);
      LLVM_DEBUG(llvm::dbgs() << "\n");
      assert(!SymbolsMap.count(Name) && "TODO: Error handle.");
      auto PIndex = new ParamIndexSymbol(Name, Value);
      // For later use
      SymbolsMap[PIndex->getName()] = PIndex;
      // For resource release
      ParamIndices.emplace_back(PIndex);
    }
    if (Value >= 0) {
      // Const Index Define
      LLVM_DEBUG(llvm::dbgs() << "Add const index: " << Value << "\n");
      std::string ConstName = "ci#" + std::to_string(Value);
      if (!SymbolsMap.count(ConstName)) {
        auto PIndex = new ParamIndexSymbol(ConstName, Value);
        SymbolsMap[PIndex->getName()] = PIndex;
        ParamIndices.emplace_back(PIndex);
      }
    }
  }
  // Done: 12-08
  auto getParamIndex(StringRef Name) {
    auto Sym = getSymbolOfName(Name);
    assert(Sym && llvm::isa<ParamIndexSymbol>(Sym));
    return llvm::cast<ParamIndexSymbol>(Sym);
  }
  // Done: 12-08
  auto getParamIndex(int64_t Value) {
    assert(Value >= 0);
    std::string ConstName = "ci#" + std::to_string(Value);
    if (!SymbolsMap.count(ConstName)) {
      addParamIndexDefine(Value);
    }
    return getParamIndex(ConstName);
  }

  void addConstIndex(StringRef Name, int64_t Value) {
    LLVM_DEBUG(llvm::dbgs() << "Add const index: " << Name);
    LLVM_DEBUG(if (Value >= 0) llvm::dbgs() << " = " << Value;);
    LLVM_DEBUG(llvm::dbgs() << "\n");
    assert(!SymbolsMap.count(Name) && "TODO: Error handle.");
    auto CIS = new ParamIndexSymbol(Name, Value);
    SymbolsMap[CIS->getName()] = CIS;
    ParamIndices.emplace_back(std::unique_ptr<ParamIndexSymbol>(CIS));
  }
  SizePtr addImplicitSize() {
    SizeID ID = SizeIDSymbolMap.size();
    auto SS = new SizeSymbol(ID);
    SizeIDSymbolMap.emplace_back(SS);
    return SS;
  }
  SizePtr addSize(StringRef Name, int64_t UB = 0, int64_t LB = 0) {
    assert(!SymbolsMap.count(Name));
    SizeID ID = SizeIDSymbolMap.size();
    auto SS = new SizeSymbol(ID, Name, UB, LB);
    SizeIDSymbolMap.emplace_back(SS);
    SymbolsMap[SS->getName()] = SS;
    return SS;
  }
  // Add Tensor
  InTensorPtr addInputTensor(StringRef Name, ShapeTy Shape) {
    assert(!SymbolsMap.count(Name));
    assert(!InTensorsMap.count(Name));
    auto IT = new InputTensor(Name, Shape);
    InputTensors.emplace_back(IT);
    SymbolsMap[IT->getName()] = IT;
    InTensorsMap[IT->getName()] = IT;
    return IT;
  }
  InTensorPtr addInputTensor(StringRef Name, int Dim) {
    assert(!SymbolsMap.count(Name));
    assert(!InTensorsMap.count(Name));
    SizeID ID = SizeIDSymbolMap.size();
    ShapeTy Shape;
    for (int Axis = 0; Axis < Dim; Axis++) {
      SizeIDSymbolMap.emplace_back(new SizeSymbol(ID++));
      Shape.emplace_back(SizeIDSymbolMap.back().get());
    }
    InputTensors.emplace_back(new InputTensor(Name, Shape));
    auto IT = InputTensors.back().get();
    SymbolsMap[IT->getName()] = IT;
    InTensorsMap[IT->getName()] = IT;
    return IT;
  }
  void addContractionTree(StringRef Name, NodePtr Root) {
    CTrees.emplace_back(new ContractionTree(Name, std::move(Root)));
    auto CT = CTrees.back().get();
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
        Sz1[i] = Sz2[i];
      } else if (!Sz2[i]) {
        Sz2[i] = Sz1[i];
      } else {
        SizeSymbol::makeSizeEqual(*Sz1[i], *Sz2[i]);
      }
    }
  }
  void resolveInputSizes() {
    SizeID UID = 0;
    for (auto &SizeSym : SizeIDSymbolMap) {
      if (SizeSym->isFirstAppear()) {
        SizeSym->setUID(UID++);
      }
    }
    NumInputSizes = UID;
  }
  int64_t getInputSizeOf(SizePtr Sz) { return Sz->getUID(); }

  StringRef getName() { return Name; }
  InTensorsTy &getInputTensors() { return InputTensors; }
  CTreePtrsTy &getOutputTensors() { return OutputCTrees; }
  CTreesTy &getContractionTrees() { return CTrees; }

  // TODO:

  int getNumInputSizs() { return NumInputSizes; }

  int getNumInputTensors() { return InputTensors.size(); }
  int getNumOutputTensors() { return OutputCTrees.size(); }
};

#undef DEBUG_TYPE

} // namespace contraction

#endif