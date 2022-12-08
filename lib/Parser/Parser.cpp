#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "TCC/Parser.h"

#define DEBUG_TYPE "TCParser"

using namespace llvm;

namespace contraction {
namespace parser {
enum Token : int {
  Token_Comma = ',',
  Token_ParentheseOpen = '(',
  Token_ParentheseClose = ')',
  Token_Equal = '=',
  Token_Addition = '+',
  Token_Multiplication = '*',
  Token_NewLine = '\n',
  Token_EOF = -1,
  Token_Contraction = -2,
  Token_Keep = -3,
  Token_Identifier = -4,
  Token_Index = -5,
  Token_Number = -6,
  Token_Input = -7,
  Token_Negative = -8,
  Token_Size = -9
};

class Lexer {
  std::unique_ptr<llvm::MemoryBuffer> MemoryBuffer;
  StringRef Buffer;
  int LastChar;
  size_t Position;
  std::string IdOrNum;

  inline int getNextChar() {
    if (Position >= Buffer.size())
      return EOF;
    return Buffer[Position++];
  }

  void setNumberStr(bool Pos = true) {
    IdOrNum = Pos ? "" : "-";
    do {
      IdOrNum += (char)LastChar;
      LastChar = getNextChar();
    } while (isdigit(LastChar) || LastChar == '.');
  }

public:
  Lexer(std::unique_ptr<llvm::MemoryBuffer> MemoryBuffer) {
    Buffer = MemoryBuffer->getBuffer();
    this->MemoryBuffer = std::move(MemoryBuffer);
    Position = 0;
    LastChar = ' ';
  }

  Token getNextToken() {
    while (LastChar != '\n' && isspace(LastChar))
      LastChar = getNextChar();
    if (isalpha(LastChar)) {
      IdOrNum = "";
      do {
        IdOrNum += (char)LastChar;
        LastChar = getNextChar();
      } while (isalnum(LastChar) || LastChar == '_');
      if (IdOrNum == "sum")
        return Token_Contraction;
      if (IdOrNum == "keep")
        return Token_Keep;
      if (IdOrNum == "index")
        return Token_Index;
      if (IdOrNum == "size")
        return Token_Size;
      if (IdOrNum == "input")
        return Token_Input;
      return Token_Identifier;
    }
    if (LastChar == '-') {
      LastChar = getNextChar();
      if (isdigit(LastChar)) {
        setNumberStr(false);
        return Token_Number;
      } else {
        return Token_Negative;
      }
    }
    if (isdigit(LastChar)) {
      setNumberStr();
      return Token_Number;
    }
    Token T{LastChar};
    LastChar = getNextChar();
    return T;
  }
  APFloat getAPFloat() { return APFloat(APFloat::IEEEdouble(), IdOrNum); }
  APInt getAPInt() { return APInt(64, IdOrNum, 10); }
  std::string getIdentifier() { return IdOrNum; }
};

using NamesTy = SmallVector<std::string, 4>;
class Parser {

  Lexer &Lxr;
  Token CurToken;

  std::unique_ptr<ContractionForest> CF;

  // Indexes
  IndexID IID;
  DenseMap<StringRef, IndexPtr> IndexesMap;

  inline bool isNextLineToken() {
    return CurToken == Token_NewLine || CurToken == Token_EOF;
  }

  void parseLine();
  void parseSizeDefines();
  void parseInputDefines();
  void parseParamIndexDefines();
  void parseContractionTree();
  void parseKeepTensors();
  IndexUsesTy parseIndexUses();
  IndexDefsTy parseIterIndexDefines();
  void eraseIterIndexDefines(IndexDefsTy &IndexDefs);
  NamesTy parseIndexNames();
  NodePtr parseAddition();
  NodePtr parseSingleNode();
  NodePtr parseNegative();
  NodePtr parseMultiplication();
  NodePtr parseIndexedTensor();
  NodePtr parseContraction();

  void inferenceIndexSizes(ShapeTy &Shape, IndexUsesTy &IndexUses) {
    assert(Shape.size() == IndexUses.size() && "TODO: Error handle.");
    for (auto Pack : llvm::zip(Shape, IndexUses)) {
      auto Size = std::get<0>(Pack);
      auto Index = std::get<1>(Pack);
      if (Index->isImplicit())
        continue;
      if (Index->getSize()) {
        SizeSymbol::makeSizeEqual(*Size, *(Index->getSize()));
      } else {
        Index->setSize(Size);
      }
    }
  }

public:
  Parser(Lexer &Lxr) : Lxr(Lxr) {}
  std::unique_ptr<ContractionForest> parseContractionForest();
};

std::unique_ptr<ContractionForest> Parser::parseContractionForest() {
  CF = std::make_unique<ContractionForest>();
  CurToken = Lxr.getNextToken();
  while (CurToken != Token_EOF) {
    parseLine();
    CurToken = Lxr.getNextToken();
  }
  // TODO: After Parse
  CF->resolveInputSizes();
  return std::move(CF);
}

void Parser::parseLine() {
  if (CurToken == Token_Keep)
    parseKeepTensors();
  else if (CurToken == Token_Identifier)
    parseContractionTree();
  else if (CurToken == Token_Index)
    parseParamIndexDefines();
  else if (CurToken == Token_Size)
    parseSizeDefines();
  else if (CurToken == Token_Input)
    parseInputDefines();
  else if (!isNextLineToken())
    llvm_unreachable("TODO: Error handle.");
  assert(isNextLineToken());
}
void Parser::parseSizeDefines() {
  LLVM_DEBUG(dbgs() << "Parse size line\n");
  CurToken = Lxr.getNextToken();
  while (CurToken == Token_Identifier) {
    auto Name = Lxr.getIdentifier();
    int64_t UB = 0, LB = 0;
    CurToken = Lxr.getNextToken();
    if (CurToken == Token_Equal) {
      CurToken = Lxr.getNextToken();
      assert(CurToken == Token_ParentheseOpen && "TODO: Error handle.");
      CurToken = Lxr.getNextToken();
      if (CurToken == Token_Number) {
        LB = Lxr.getAPInt().getSExtValue();
        CurToken = Lxr.getNextToken();
      }
      assert(CurToken == Token_Comma && "TODO: Error handle.");
      CurToken = Lxr.getNextToken();
      if (CurToken == Token_Number) {
        UB = Lxr.getAPInt().getSExtValue();
        CurToken = Lxr.getNextToken();
      }
      assert(CurToken == Token_ParentheseClose && "TODO: Error handle.");
      CurToken = Lxr.getNextToken();
    }
    CF->addSizeDefine(Name, UB, LB);
    if (CurToken == Token_Comma)
      CurToken = Lxr.getNextToken();
    else
      break;
  }
}

// Done 12-08
void Parser::parseParamIndexDefines() {
  LLVM_DEBUG(dbgs() << "Parse index line\n");
  CurToken = Lxr.getNextToken();
  while (CurToken == Token_Identifier) {
    auto Name = Lxr.getIdentifier();
    int64_t Value = ParamIndexSymbol::Undefined;
    CurToken = Lxr.getNextToken();
    if (CurToken == Token_Equal) {
      CurToken = Lxr.getNextToken();
      assert(CurToken == Token_Number && "TODO: Error handle.");
      Value = Lxr.getAPInt().getSExtValue();
      assert(Value >= 0 && "TODO: Error handle.");
      CurToken = Lxr.getNextToken();
    }
    CF->addParamIndexDefine(Value, Name);
    if (CurToken == Token_Comma)
      CurToken = Lxr.getNextToken();
    else
      break;
  }
}

void Parser::parseInputDefines() {
  LLVM_DEBUG(dbgs() << "Parse input line\n");
  CurToken = Lxr.getNextToken();
  while (CurToken == Token_Identifier) {
    auto Name = Lxr.getIdentifier();
    CurToken = Lxr.getNextToken();
    ShapeTy Shape;
    if (CurToken == Token_ParentheseOpen) {
      CurToken = Lxr.getNextToken();
      while (true) {
        if (CurToken == Token_Identifier) {
          auto Name = Lxr.getIdentifier();
          if (auto Sym = CF->getSymbolOfName(Name)) {
            auto Size = cast<SizeSymbol>(Sym);
            Shape.emplace_back(Size);
          } else {
            Shape.emplace_back(CF->addSizeDefine(Name));
          }
          CurToken = Lxr.getNextToken();
        } else if (CurToken == Token_Multiplication) {
          Shape.emplace_back(CF->addSizeDefine());
          CurToken = Lxr.getNextToken();
        } else {
          break;
        }
        if (CurToken == Token_Comma)
          CurToken = Lxr.getNextToken();
        else
          break;
      }
      assert(CurToken == Token_ParentheseClose);
      CurToken = Lxr.getNextToken();
    }
    CF->addInputTensorDefine(Name, Shape);
    if (CurToken == Token_Comma)
      CurToken = Lxr.getNextToken();
    else
      break;
  }
}

void Parser::parseContractionTree() {
  // get Name
  auto Name = std::string(Lxr.getIdentifier());
  CurToken = Lxr.getNextToken();
  // parse Indexes
  IID = 0;
  IndexDefsTy IndexDefs = parseIterIndexDefines();
  // parse expressions
  assert(CurToken == Token_Equal);
  CurToken = Lxr.getNextToken();
  auto Child = parseAddition();
  eraseIterIndexDefines(IndexDefs);
  // create tree
  auto Root = RootNode::get(IndexDefs, std::move(Child));
  CF->addContractionTree(Name, std::move(Root));
}

NamesTy Parser::parseIndexNames() {
  NamesTy Names;
  if (CurToken != Token_ParentheseOpen)
    return Names;
  CurToken = Lxr.getNextToken();
  while (CurToken == Token_Identifier) {
    Names.emplace_back(Lxr.getIdentifier());
    CurToken = Lxr.getNextToken();
    if (CurToken == Token_Comma)
      CurToken = Lxr.getNextToken();
    else
      break;
  }
  assert(CurToken == Token_ParentheseClose && "TODO: Error handle.");
  CurToken = Lxr.getNextToken();
  return Names;
}

NodePtr Parser::parseAddition() {
  SmallVector<bool> AddOrSub(1, true);
  ChildNodesTy Childs;
  while (true) {
    Childs.emplace_back(parseMultiplication());
    if (CurToken == Token_Addition) {
      AddOrSub.emplace_back(true);
      CurToken = Lxr.getNextToken();
    } else if (CurToken == Token_Negative) {
      AddOrSub.emplace_back(false);
      CurToken = Lxr.getNextToken();
      assert(CurToken != Token_Negative &&
             "TODO: double negative, too complex.");
    } else {
      break;
    }
  }
  if (Childs.size() == 1)
    return std::move(Childs[0]);
  return AdditionNode::get(Childs, AddOrSub);
}

NodePtr Parser::parseSingleNode() {
  if (CurToken == Token_Contraction) {
    return parseContraction();
  } else if (CurToken == Token_Negative) {
    return parseNegative();
  } else if (CurToken == Token_ParentheseOpen) {
    CurToken = Lxr.getNextToken();
    auto Child = parseAddition();
    assert(CurToken == Token_ParentheseClose);
    CurToken = Lxr.getNextToken();
    return Child;
  } else if (CurToken == Token_Identifier) {
    return parseIndexedTensor();
  } else if (CurToken == Token_Number) {
    auto Value = Lxr.getAPFloat().convertToDouble();
    CurToken = Lxr.getNextToken();
    return ConstantNode::get(Value);
  } else {
    llvm_unreachable("TODO: Expression should begin with sum or identifier or "
                     "open parenthese or negative.");
  }
}

NodePtr Parser::parseNegative() {
  assert(CurToken == Token_Negative);
  CurToken = Lxr.getNextToken();
  assert(CurToken != Token_Negative);
  auto Child = parseSingleNode();
  return NegativeNode::get(std::move(Child));
}

NodePtr Parser::parseMultiplication() {
  ChildNodesTy Childs;
  while (true) {
    Childs.emplace_back(parseSingleNode());
    if (CurToken != Token_Multiplication)
      break;
    CurToken = Lxr.getNextToken();
  }
  if (Childs.size() == 1)
    return std::move(Childs[0]);
  return MultiplicationNode::get(Childs);
}

// Done 12-08
IndexDefsTy Parser::parseIterIndexDefines() {
  IndexDefsTy IndexDefs;
  NamesTy Names;
  if (CurToken != Token_ParentheseOpen)
    return IndexDefs;
  CurToken = Lxr.getNextToken();
  while (CurToken == Token_Identifier) {
    auto Name = Lxr.getIdentifier();
    assert(!IndexesMap.count(Name) && "TODO: Error handle.");
    auto IIndex = new IterIndexSymbol(IID++, Name);
    IndexesMap[IIndex->getName()] = IIndex;
    IndexDefs.emplace_back(IIndex);
    CurToken = Lxr.getNextToken();
    if (CurToken == Token_Comma)
      CurToken = Lxr.getNextToken();
    else
      break;
  }
  assert(CurToken == Token_ParentheseClose && "TODO: Error handle.");
  CurToken = Lxr.getNextToken();
  return IndexDefs;
}
void Parser::eraseIterIndexDefines(IndexDefsTy &IndexDefs) {
  for (auto &IndexDef : IndexDefs) {
    assert(IndexesMap.erase(IndexDef->getName()));
  }
}

// Done 12-08
NodePtr Parser::parseContraction() {
  assert(CurToken == Token_Contraction);
  CurToken = Lxr.getNextToken();
  assert(CurToken == Token_ParentheseOpen);
  CurToken = Lxr.getNextToken();
  // parse Indexes
  IndexDefsTy IndexDefs = parseIterIndexDefines();
  CurToken = Lxr.getNextToken();
  // parse expression
  auto child = parseAddition();
  assert(CurToken == Token_ParentheseClose);
  CurToken = Lxr.getNextToken();
  eraseIterIndexDefines(IndexDefs);
  // create Node
  return ContractionNode::get(IndexDefs, std::move(child));
}

// Done 12-08
IndexUsesTy Parser::parseIndexUses() {
  IndexUsesTy IndexUses;
  if (CurToken != Token_ParentheseOpen)
    return IndexUses;
  CurToken = Lxr.getNextToken();
  while (true) {
    if (CurToken == Token_Identifier) {
      // Iter index
      auto Name = Lxr.getIdentifier();
      if (IndexesMap.count(Name)) {
        IndexUses.emplace_back(IndexesMap[Name]);
      } else {
        // Param Index
        auto PIndex = CF->getParamIndex(Name);
        IndexUses.emplace_back(PIndex);
      }
    } else if (CurToken == Token_Number) {
      // Const index
      auto Value = Lxr.getAPInt().getSExtValue();
      assert(Value >= 0);
      auto PIndex = CF->getParamIndex(Value);
      IndexUses.emplace_back(PIndex);
    } else
      break;
    CurToken = Lxr.getNextToken();
    if (CurToken == Token_Comma)
      CurToken = Lxr.getNextToken();
    else
      break;
  }
  assert(CurToken == Token_ParentheseClose && "TODO: Error handle.");
  CurToken = Lxr.getNextToken();
  return IndexUses;
}

// Done 12-08
NodePtr Parser::parseIndexedTensor() {
  // Parse tensor
  auto Name = std::string(Lxr.getIdentifier());
  CurToken = Lxr.getNextToken();
  // Parse indices
  auto IndexUses = parseIndexUses();
  auto T = CF->getSymbolOfName(Name);
  if (auto CT = llvm::dyn_cast_if_present<ContractionTree>(T)) {
    // Parse contracted tensor
    inferenceIndexSizes(CT->getShape(), IndexUses);
    return TreeAsLeafNode::get(IndexUses, CT);
  } else {
    // Parse input tensor
    assert(!T || llvm::isa<InputTensor>(T));
    auto IT = llvm::cast_if_present<InputTensor>(T);
    if (!IT) {
      IT = CF->addInputTensorDefine(Name, IndexUses.size());
    }
    inferenceIndexSizes(IT->getShape(), IndexUses);
    return InputAsLeafNode::get(IndexUses, IT);
  }
}

void Parser::parseKeepTensors() {
  assert(CurToken == Token_Keep);
  CurToken = Lxr.getNextToken();
  while (CurToken == Token_Identifier) {
    auto Name = Lxr.getIdentifier();
    CF->addKeep(Name);
    CurToken = Lxr.getNextToken();
    if (CurToken != Token_Comma)
      break;
    CurToken = Lxr.getNextToken();
  }
  assert(isNextLineToken());
  CurToken = Lxr.getNextToken();
}
} // namespace parser

std::unique_ptr<ContractionForest> parseInputFile(StringRef FileName) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> fileOrErr =
      MemoryBuffer::getFileOrSTDIN(FileName);
  if (std::error_code EC = fileOrErr.getError()) {
    errs() << "Open input file failed: " << EC.message() << "\n";
    return nullptr;
  }
  parser::Lexer Lexer{std::move(fileOrErr.get())};
  parser::Parser Parser{Lexer};
  auto CF = Parser.parseContractionForest();
  return CF;
}
} // namespace contraction

#undef DEBUG_TYPE
