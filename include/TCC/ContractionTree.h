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
using InputTensorPtr = InputTensor *;

// ContractionTree - Line of the Contraction Expression, which do
// some calculation (contraction is major) and store value to a tensor. The
// result can be used by succeed Expression.
class ContractionTree;
using TreeTensorsTy = SmallVector<std::unique_ptr<ContractionTree>>;
using CTreePtr = ContractionTree *;

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
  static const std::string implicitPrefix;
  SizeSymbol(SizeID ID) : Symbol(SK_Size, ""), ID(ID) {
    name = implicitPrefix + std::to_string(ID);
    EqualityInfo = nullptr;
  }
  SizeSymbol(SizeID ID, StringRef name, int64_t UB = 0, int64_t LB = 0)
      : Symbol(SK_Size, name), ID(ID) {
    EqualityInfo = nullptr;
    setUpperBound(UB);
    setLowerBound(LB);
  }
  // Member Functions
  bool isImplicit() { return name.substr(0, 3) == implicitPrefix; }
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
      getInfo()->name = implicitPrefix + std::to_string(UID);
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
  const static std::string implicitPrefix;
  // Member Functions
  /// isImplicit means the index is used as A(i,3),
  /// where the second index symbol "ci#3"
  bool isImplicit() { return getName().substr(0, 3) == implicitPrefix; }
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
  InputTensorPtr InputRef;
  InputTensorNode(IndexUsesTy &IndexUses, InputTensorPtr InputRef)
      : LeafNode(NK_InputTensor, IndexUses), InputRef(InputRef) {}

public:
  static NodePtr get(IndexUsesTy &IndexUses, InputTensorPtr InputRef) {
    return NodePtr(new InputTensorNode(IndexUses, InputRef));
  }
  // Member Functions
  InputTensorPtr getInput() { return InputRef; }
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
  // Although ContractionTree is a tensor symbol, it's shape stores in
  // root node, but can use trait to make it works as tensor.
  // It's not important.
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
  /// Symbol Map
  using SymbolMapTy = DenseMap<StringRef, SymbolPtr>;
  SymbolMapTy symbolMap;
  using InputTensorsMapTy = DenseMap<StringRef, InputTensorPtr>;
  InputTensorsMapTy inputTensorMap;
  /// Context, hold global resources
  // Global indices
  ParamIndicesHoldTy paramIndices;
  // Sizes
  SizesHoldTy sizeIDSymbolMap;
  // Input tensors
  InputTensorsHoldTy inputTensors;
  // Calculated tensors
  TreeTensorsTy treeTensors;
  /// Function argument
  using InputTensorPtrsTy = SmallVector<InputTensorPtr>;
  /// Function results
  using TreeTensorPtrsTy = SmallVector<CTreePtr>;
  TreeTensorPtrsTy resultTensors;
  InputTensorPtrsTy keepedInputTensors;

  // Contraction Tree Tensors
  // For resource release

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

  // Modified 23-03-28
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
    if (symbolMap.count(name)) {
      auto symbol = symbolMap[name];
      assert(llvm::isa<T>(symbol));
      return llvm::cast<T>(symbol);
    }
    return nullptr;
  }
  template <typename T> T *getSymbolOrNull(StringRef name) {
    if (symbolMap.count(name)) {
      auto symbol = symbolMap[name];
      return llvm::dyn_cast_or_null<T>(symbol);
    }
    return nullptr;
  }
  // For indices
  void insertParamIndex(ParamIndexPtr index) {
    insertSymbol(index);
    paramIndices.emplace_back(index);
  }
  void addParamIndexDefine(StringRef name, int64_t value) {
    LLVM_DEBUG(llvm::dbgs() << "Add param index: " << name);
    LLVM_DEBUG(if (value >= 0) llvm::dbgs() << " = " << value);
    LLVM_DEBUG(llvm::dbgs() << "\n");
    assert(!symbolMap.count(name) && "TODO: Error handle.");
    auto index = new ParamIndexSymbol(name, value);
    insertParamIndex(index);
  }
  auto getImplicitParamIndex(int64_t value) {
    assert(value >= 0);
    std::string name = IndexSymbol::implicitPrefix + std::to_string(value);
    if (auto index = getSymbolIfPresent<ParamIndexSymbol>(name))
      return index;
    auto index = new ParamIndexSymbol(name, value);
    insertParamIndex(index);
    return index;
  }
  // For sizes
  SizePtr addSizeDefine() {
    SizeID sizeID = sizeIDSymbolMap.size();
    auto size = new SizeSymbol(sizeID);
    sizeIDSymbolMap.emplace_back(size);
    return size;
  }
  SizePtr addSizeDefine(StringRef name, int64_t UB = 0, int64_t LB = 0) {
    assert(!symbolMap.count(name));
    SizeID sizeID = sizeIDSymbolMap.size();
    auto size = new SizeSymbol(sizeID, name, UB, LB);
    sizeIDSymbolMap.emplace_back(size);
    insertSymbol(size);
    return size;
  }
  // For input tensors
  void addInputTensorDefine(StringRef name, SizesTy shape) {
    assert(!symbolMap.count(name));
    assert(!inputTensorMap.count(name));
    auto inputTensor = new InputTensor(name, shape);
    inputTensors.emplace_back(inputTensor);
    inputTensorMap[inputTensor->getName()] = inputTensor;
    insertSymbol(inputTensor);
  }
  InputTensorPtr getOrAddInputTensorDefine(StringRef name, int rank) {
    if (auto inputTensor = getSymbolIfPresent<InputTensor>(name))
      return inputTensor;
    SizesTy shape;
    for (int i = 0; i < rank; i++) {
      shape.emplace_back(addSizeDefine());
    }
    addInputTensorDefine(name, shape);
    return getSymbol<InputTensor>(name);
  }
  // For tree Tensors
  void addContractionTree(StringRef name, NodePtr root) {
    auto treeTensor = new ContractionTree(name, std::move(root));
    treeTensors.emplace_back(treeTensor);
    insertSymbol(treeTensor);
  }
  // For keep statement
  void addTensorKeeped(StringRef name) {
    if (auto treeTensor = getSymbolOrNull<ContractionTree>(name)) {
      resultTensors.emplace_back(treeTensor);
    } else if (auto inputTensor = getSymbolOrNull<InputTensor>(name)) {
      keepedInputTensors.emplace_back(inputTensor);
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
    for (auto &Size : sizeIDSymbolMap) {
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
    for (auto &Index : paramIndices) {
      if (!Index->hasValue()) {
        ArgIndices.emplace_back(Index.get());
      } else {
        ConstIndices.emplace_back(Index.get());
      }
    }
    for (auto &InTensor : inputTensors) {
      ArgTensors.emplace_back(InTensor.get());
    }
  }
  int64_t getInputSizeOf(SizePtr Sz) { return Sz->getUID(); }

  StringRef getName() { return name; }
  auto &getInputTensors() { return inputTensors; }
  auto &getOutputTensors() { return resultTensors; }
  TreeTensorsTy &getContractionTrees() { return treeTensors; }

  auto &getArgSizes() { return ArgSizes; }
  auto &getArgIndices() { return ArgIndices; }
  auto &getArgTensors() { return inputTensors; }
  auto &getConstIndices() { return ConstIndices; }
  auto &getRetTensors() { return resultTensors; }

  // TODO:

  int getNumInputSizs() { return ArgSizes.size(); }

  int getNumInputTensors() { return inputTensors.size(); }
  int getNumOutputTensors() { return resultTensors.size(); }
};

#undef DEBUG_TYPE

} // namespace tc

#endif