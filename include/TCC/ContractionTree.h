#ifndef TCC_CONTRACTIONTREE_H
#define TCC_CONTRACTIONTREE_H

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

namespace tc {
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

// InputTensor - The arrays used by Contraction Expression that not
// defined by predecessor expressions.
class InputTensor;
using InTensorsTy = SmallVector<std::unique_ptr<InputTensor>>;
using InTensorPtr = InputTensor *;
using InputTensorPtrsTy = SmallVector<InTensorPtr>;
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
  std::string name;
  Symbol(SymbolKind K, StringRef name) : Kind(K), name(name) {}

public:
  // Member Functions
  StringRef getName() { return name; }
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
using SizesTy = SizesTy;
// SizeID map to SizeSymbol
using SizesMapTy = SmallVector<std::unique_ptr<SizeSymbol>>;
class SizeSymbol : public Symbol {
private:
  SizeID ID;

  class EqualityStorage {
  public:
    std::string name;
    SizeID ID;
    SizeID UID = -1;
    int64_t UpperBound = 0;
    int64_t LowerBound = 0;
    SizesTy Aliases;
    EqualityStorage(ArrayRef<SizePtr> Aliases) : Aliases(Aliases) {}
    void setID(SizeID ID) { this->ID = this->ID < ID ? this->ID : ID; }
    void setLowerBound(int64_t LB) {
      LowerBound = LowerBound > LB ? LowerBound : LB;
    }
    void setUpperBound(int64_t UB) {
      UpperBound = UB && (UpperBound > UB || !UpperBound) ? UB : UpperBound;
    }
  };
  EqualityStorage *EqualityInfo;

  EqualityStorage *getInfo() {
    if (!EqualityInfo) {
      EqualityInfo = new EqualityStorage{{this}};
      EqualityInfo->ID = ID;
      if (!isImplicit())
        EqualityInfo->name = name;
    }
    return EqualityInfo;
  }

public:
  SizeSymbol(SizeID ID) : Symbol(SK_Size, ""), ID(ID) {
    name = "sz#" + std::to_string(ID);
    EqualityInfo = nullptr;
  }
  SizeSymbol(SizeID ID, StringRef name, int64_t UB = 0, int64_t LB = 0)
      : Symbol(SK_Size, name), ID(ID) {
    EqualityInfo = nullptr;
    setUpperBound(UB);
    setLowerBound(LB);
  }
  // Member Functions
  bool isImplicit() { return name.substr(0, 3) == "sz#"; }
  bool operator==(const SizeSymbol &Sz) {
    return EqualityInfo && EqualityInfo == Sz.EqualityInfo;
  }
  static void makeSizeEqual(SizeSymbol &Sz1, SizeSymbol &Sz2) {
    if (Sz1 == Sz2)
      return;
    if (!Sz1.EqualityInfo && Sz2.EqualityInfo) {
      return makeSizeEqual(Sz2, Sz1);
    } else if (!Sz2.EqualityInfo) {
      auto Info = Sz1.getInfo();
      Info->Aliases.emplace_back(&Sz2);
      Sz2.EqualityInfo = Info;
      Info->setID(Sz2.ID);
      if (Info->name.empty() && !Sz2.isImplicit()) {
        Info->name = Sz2.name;
      }
    } else {
      auto Info1 = Sz1.EqualityInfo;
      auto Info2 = Sz2.EqualityInfo;
      Info1->setID(Info2->ID);
      Info1->setUpperBound(Info2->UpperBound);
      Info1->setLowerBound(Info2->LowerBound);
      if (Info1->name.empty()) {
        Info1->name = Info2->name;
      }
      for (auto Alias : Sz2.EqualityInfo->Aliases) {
        Info1->Aliases.emplace_back(Alias);
        Alias->EqualityInfo = Info1;
      }
      delete Info2;
    }
  }
  void setUpperBound(int64_t UB) { getInfo()->UpperBound = UB; }
  void setLowerBound(int64_t LB) { getInfo()->LowerBound = LB; }
  bool isFirstAppear() { return ID == getInfo()->ID; }
  void setUID(SizeID UID) {
    getInfo()->UID = UID;
    if (getInfo()->name.empty())
      getInfo()->name = "sz#" + std::to_string(UID);
  }
  StringRef getUName() { return getInfo()->name; }
  auto getID() { return getInfo()->ID; }
  auto getUID() { return getInfo()->UID; }
  auto getUpperBound() { return getInfo()->UpperBound; }
  auto getLowerBound() { return getInfo()->LowerBound; }
  auto isConstant() {
    return getUpperBound() && getUpperBound() == getLowerBound();
  }
  auto getAliases() { return getInfo()->Aliases; }
  // For LLVM RTTI
  static bool classof(const Symbol *S) { return S->getKind() == SK_Size; }
  // Print
  bool hasAliases() {
    auto Aliases = getInfo()->Aliases;
    for (auto Alias : Aliases) {
      if (Alias == this)
        continue;
      if (!Alias->isImplicit())
        return true;
    }
    return false;
  }
  void dump();
  void dumpAliases();
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, SizePtr Size) {
    if (Size->isImplicit() && Size->getUID() >= 0) {
      OS << Size->getUName();
    } else {
      OS << Size->getName();
    }
    return OS;
  }
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
  IndexSymbol(SymbolKind K, StringRef name) : Symbol(K, name), Size(nullptr) {}

public:
  // Member Functions
  /// isImplicit means the index is used as A(i,3),
  /// where the second index symbol "ci#3"
  bool isImplicit() { return getName().substr(0, 3) == "ci#"; }
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
  IterIndexSymbol(IndexID ID, StringRef name)
      : IndexSymbol(SK_IterIndex, name), ID(ID) {}
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
  ParamIndexSymbol(StringRef name, int64_t Value)
      : IndexSymbol(SK_ParamIndex, name), Value(Value) {}
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
  SizesTy shape;

public:
  InputTensor(StringRef name, SizesTy Shape)
      : Symbol(SK_InTenser, name), shape(Shape) {}
  SizesTy &getShape() { return shape; }
  int64_t getDimension() { return shape.size(); }
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
    NK_TreeTensor,
    NK_InputTensor
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
  SizesTy Shape;
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
  SizesTy &getShape() { return Shape; }
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
public:
  using SignsTy = SmallVector<bool>;

private:
  SignsTy AddOrSub;
  AdditionNode(ChildNodesTy &Childs, SmallVector<bool> &AddOrSub)
      : ArithNode(NK_Addition, Childs), AddOrSub(AddOrSub) {}

public:
  static NodePtr get(ChildNodesTy &Childs, SignsTy &AddOrSub) {
    return NodePtr(new AdditionNode(Childs, AddOrSub));
  }
  auto &getAddOrSub() { return AddOrSub; }
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
    return N->getKind() == NK_TreeTensor || N->getKind() == NK_InputTensor;
  }
};

// TreeTensorNode: Reference to a contraction expression result.
class TreeTensorNode : public LeafNode {
private:
  CTreePtr TreeRef;
  TreeTensorNode(IndexUsesTy &IndexUses, CTreePtr TreeRef)
      : LeafNode(NK_TreeTensor, IndexUses), TreeRef(TreeRef) {}

public:
  static NodePtr get(IndexUsesTy &IndexUses, CTreePtr TreeRef) {
    return NodePtr(new TreeTensorNode(IndexUses, TreeRef));
  }
  // Member Functions
  CTreePtr getTree() { return TreeRef; }
  // For LLVM RTTI
  static bool classof(const Node *N) { return N->getKind() == NK_TreeTensor; }
  // dump
  void dump(int);
};

// InputTensorNode: Reference to an input tensor.
class InputTensorNode : public LeafNode {
private:
  InTensorPtr InputRef;
  InputTensorNode(IndexUsesTy &IndexUses, InTensorPtr InputRef)
      : LeafNode(NK_InputTensor, IndexUses), InputRef(InputRef) {}

public:
  static NodePtr get(IndexUsesTy &IndexUses, InTensorPtr InputRef) {
    return NodePtr(new InputTensorNode(IndexUses, InputRef));
  }
  // Member Functions
  InTensorPtr getInput() { return InputRef; }
  // For LLVM RTTI
  static bool classof(const Node *N) { return N->getKind() == NK_InputTensor; }
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
  // Member Function
  Node &getChild() { return *Childs[0]; }
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
  // Member Functions
  auto getValue() { return Value; }
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
  } else if (auto *N = llvm::dyn_cast<TreeTensorNode>(this)) {
    delete N;
  } else if (auto *N = llvm::dyn_cast<InputTensorNode>(this)) {
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
  ContractionTree(StringRef name, NodePtr Root)
      : Symbol(SK_CTensor, name), Root(std::move(Root)) {
    assert(llvm::isa<RootNode>(this->Root.get()));
  }
  RootNode &getRootNode() { return llvm::cast<RootNode>(*Root); }
  SizesTy &getShape() { return getRootNode().getShape(); }
  int64_t getDimension() { return getShape().size(); }
  // For LLVM RTTI
  static bool classof(const Symbol *S) { return S->getKind() == SK_CTensor; }
  // dump
  void dump();
};

#define DEBUG_TYPE "tcc-ast"

using SizesHoldTy = SmallVector<std::unique_ptr<SizeSymbol>>;
using ParamIndicesHoldTy = SmallVector<std::unique_ptr<ParamIndexSymbol>>;
using InputTensorsHoldTy = SmallVector<std::unique_ptr<InputTensor>>;
using SizePtrsTy = SmallVector<SizePtr>;
using ParamIndexPtrsTy = SmallVector<ParamIndexSymbol *>;
class ContractionForest {
  std::string name;
  // Symbols
  using SymbolMapTy = DenseMap<StringRef, SymbolPtr>;
  SymbolMapTy symbolMap;
  // Const Indexes
  // Input Tensors

  InTensorsMapTy InTensorsMap;
  InputTensorPtrsTy ReservedInputs;
  // Contraction Tree Tensors
  CTreesTy CTrees;
  CTreesMapTy CTreesMap;
  CTreePtrsTy OutputCTrees;
  // For resource release
  SizesHoldTy SizeIDSymbolMap;
  ParamIndicesHoldTy ParamIndices;
  InputTensorsHoldTy InputTensors;
  // Function Argument
  SizePtrsTy ArgSizes;
  ParamIndexPtrsTy ArgIndices;
  InputTensorPtrsTy ArgTensors;
  // Others
  SizePtrsTy ConstSizes;
  ParamIndexPtrsTy ConstIndices;

public:
  ContractionForest() { name = "CT"; }
  void dump();

  // Done 12-08
  void insertSymbol(SymbolPtr symbol) {
    LLVM_DEBUG(llvm::dbgs() << "Insert " << symbol->getName() << "\n");
    symbolMap[symbol->getName()] = symbol;
  }
  SymbolPtr getSymbol(StringRef name) {
    return symbolMap.count(name) ? symbolMap[name] : nullptr;
  }
  template <typename T> T *getSymbol(StringRef name) {
    assert(symbolMap.count(name));
    auto symbol = symbolMap[name];
    assert(llvm::isa<T>(symbol));
    return llvm::cast<T>(symbol);
  }
  template <typename T> T *getSymbolIfPresent(StringRef name) {
    if (!symbolMap.count(name))
      return nullptr;
    auto symbol = symbolMap[name];
    assert(llvm::isa<T>(symbol));
    return llvm::cast<T>(symbol);
  }
  // Done 12-08
  // Done 12-08
  void addParamIndexDefine(int64_t Value, StringRef name = "") {
    if (!name.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "Add param index: " << name);
      LLVM_DEBUG(if (Value >= 0) llvm::dbgs() << " = " << Value);
      LLVM_DEBUG(llvm::dbgs() << "\n");
      assert(!symbolMap.count(name) && "TODO: Error handle.");
      auto PIndex = new ParamIndexSymbol(name, Value);
      // For later use
      insertSymbol(PIndex);
      // For resource release
      ParamIndices.emplace_back(PIndex);
    }
    if (Value >= 0) {
      // Const Index Define
      LLVM_DEBUG(llvm::dbgs() << "Add const index: " << Value << "\n");
      std::string ConstName = "ci#" + std::to_string(Value);
      if (!symbolMap.count(ConstName)) {
        auto PIndex = new ParamIndexSymbol(ConstName, Value);
        insertSymbol(PIndex);
        ParamIndices.emplace_back(PIndex);
      }
    }
  }
  // Done: 12-08
  auto getParamIndex(StringRef name) {
    auto Sym = getSymbol(name);
    assert(Sym && llvm::isa<ParamIndexSymbol>(Sym));
    return llvm::cast<ParamIndexSymbol>(Sym);
  }
  // Done: 12-08
  auto getParamIndex(int64_t Value) {
    assert(Value >= 0);
    std::string ConstName = "ci#" + std::to_string(Value);
    if (!symbolMap.count(ConstName)) {
      addParamIndexDefine(Value);
    }
    return getParamIndex(ConstName);
  }

  // Done 12-08
  SizePtr addSizeDefine() {
    SizeID ID = SizeIDSymbolMap.size();
    auto SS = new SizeSymbol(ID);
    SizeIDSymbolMap.emplace_back(SS);
    return SS;
  }
  SizePtr addSizeDefine(StringRef name, int64_t UB = 0, int64_t LB = 0) {
    assert(!symbolMap.count(name));
    SizeID ID = SizeIDSymbolMap.size();
    auto SS = new SizeSymbol(ID, name, UB, LB);
    SizeIDSymbolMap.emplace_back(SS);
    insertSymbol(SS);
    return SS;
  }

  // Done 12-08
  InTensorPtr addInputTensorDefine(StringRef name, SizesTy Shape) {
    assert(!symbolMap.count(name));
    assert(!InTensorsMap.count(name));
    auto IT = new InputTensor(name, Shape);
    InputTensors.emplace_back(IT);
    insertSymbol(IT);
    InTensorsMap[IT->getName()] = IT;
    return IT;
  }
  InTensorPtr addInputTensorDefine(StringRef name, int Dim) {
    SizesTy Shape;
    for (int Axis = 0; Axis < Dim; ++Axis) {
      Shape.emplace_back(addSizeDefine());
    }
    return addInputTensorDefine(name, Shape);
  }

  void addContractionTree(StringRef name, NodePtr Root) {
    CTrees.emplace_back(new ContractionTree(name, std::move(Root)));
    auto CT = CTrees.back().get();
    insertSymbol(CT);
    CTreesMap[CT->getName()] = CT;
  }
  // Add Keep
  void addTensorKeeped(StringRef name) {
    auto Sym = getSymbol(name);
    if (auto CT = llvm::dyn_cast_if_present<ContractionTree>(Sym)) {
      OutputCTrees.emplace_back(CT);
    } else if (auto IT = llvm::dyn_cast_if_present<InputTensor>(Sym)) {
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
  void finishForestConstruct() {
    for (auto &Size : SizeIDSymbolMap) {
      if (Size->isFirstAppear()) {
        if (!Size->isConstant()) {
          Size->setUID(ArgSizes.size());
          ArgSizes.emplace_back(Size.get());
        } else {
          ConstSizes.emplace_back(Size.get());
        }
      }
    }
    SizeID UID = ArgSizes.size();
    for (auto Size : ConstSizes) {
      Size->setUID(UID++);
    }
    for (auto &Index : ParamIndices) {
      if (!Index->hasValue()) {
        ArgIndices.emplace_back(Index.get());
      } else {
        ConstIndices.emplace_back(Index.get());
      }
    }
    for (auto &InTensor : InputTensors) {
      ArgTensors.emplace_back(InTensor.get());
    }
  }
  int64_t getInputSizeOf(SizePtr Sz) { return Sz->getUID(); }

  StringRef getName() { return name; }
  InTensorsTy &getInputTensors() { return InputTensors; }
  CTreePtrsTy &getOutputTensors() { return OutputCTrees; }
  CTreesTy &getContractionTrees() { return CTrees; }

  auto &getArgSizes() { return ArgSizes; }
  auto &getArgIndices() { return ArgIndices; }
  auto &getArgTensors() { return InputTensors; }
  auto &getConstIndices() { return ConstIndices; }
  auto &getRetTensors() { return OutputCTrees; }

  // TODO:

  int getNumInputSizs() { return ArgSizes.size(); }

  int getNumInputTensors() { return InputTensors.size(); }
  int getNumOutputTensors() { return OutputCTrees.size(); }
};

#undef DEBUG_TYPE

} // namespace tc

#endif