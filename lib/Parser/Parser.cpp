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
  APInt getAPInt() { return APInt(64, IdOrNum, 10); }
  std::string getIdentifier() { return IdOrNum; }
};

using NamesTy = SmallVector<std::string, 4>;
class Parser {

  Lexer &Lx;
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
  void parseIndexDefines();
  void parseContractionTree();
  void parseKeepTensors();
  NamesTy parseIndexNames();
  NodePtr parseAddition();
  NodePtr parseSingleNode();
  NodePtr parseNegative();
  NodePtr parseMultiplication();
  NodePtr parseIndexedTensor();
  NodePtr parseContraction();

  IndexesTy insertIndexDefines(NamesTy &Names, bool AllDef = true) {
    IndexesTy IndexDefs;
    for (auto &Name : Names) {
      if (auto CIS = CF->getSymbolOfName(Name)) {
        assert(!AllDef && "TODO: Error handle.");
        assert(isa<ConstIndexSymbol>(CIS) && "TODO: Error handle.");
        IndexDefs.emplace_back(cast<ConstIndexSymbol>(CIS));
      } else {
        assert(!IndexesMap.count(Name) && "TODO: Error handle.");
        auto IIS = new IterIndexSymbol(IID++, Name);
        IndexesMap[IIS->getName()] = IIS;
        IndexDefs.emplace_back(IIS);
      }
    }
    return IndexDefs;
  }
  void eraseIndexDefines(NamesTy &Names) {
    for (auto &Name : Names) {
      if (auto CIS = CF->getSymbolOfName(Name)) {
        assert(isa<ConstIndexSymbol>(CIS) && "Must be.");
        continue;
      }
      assert(IndexesMap.count(Name) && "Must count!");
      IndexesMap.erase(Name);
    }
  }
  IndexUsesTy getIndexUses(NamesTy &Names) {
    IndexUsesTy IndexUses;
    for (auto &Name : Names) {
      if (auto CIS = CF->getSymbolOfName(Name)) {
        assert(isa<ConstIndexSymbol>(CIS) && "TODO: Error handle.");
        IndexUses.emplace_back(cast<IndexSymbol>(CIS));
      } else {
        assert(IndexesMap.count(Name) && "TODO: Error handle.");
        IndexUses.emplace_back(IndexesMap[Name]);
      }
    }
    return IndexUses;
  }
  void inferenceIndexSizes(ShapeTy &Shape, IndexUsesTy &IndexUses) {
    assert(Shape.size() == IndexUses.size() && "TODO: Error handle.");
    for (auto Pack : llvm::zip(Shape, IndexUses)) {
      auto Size = std::get<0>(Pack);
      auto Index = std::get<1>(Pack);
      if (Index->getSize()) {
        SizeSymbol::makeSizeEqual(*Size, *(Index->getSize()));
      } else {
        Index->setSize(Size);
      }
    }
  }

public:
  Parser(Lexer &lexer) : Lx(lexer) {}
  std::unique_ptr<ContractionForest> parseContractionForest();
};

std::unique_ptr<ContractionForest> Parser::parseContractionForest() {
  CF = std::make_unique<ContractionForest>();
  CurToken = Lx.getNextToken();
  while (CurToken != Token_EOF) {
    parseLine();
    CurToken = Lx.getNextToken();
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
    parseIndexDefines();
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
  CurToken = Lx.getNextToken();
  while (CurToken == Token_Identifier) {
    auto Name = Lx.getIdentifier();
    int64_t UB = 0, LB = 0;
    CurToken = Lx.getNextToken();
    if (CurToken == Token_Equal) {
      CurToken = Lx.getNextToken();
      assert(CurToken == Token_ParentheseOpen && "TODO: Error handle.");
      CurToken = Lx.getNextToken();
      if (CurToken == Token_Number) {
        LB = Lx.getAPInt().getSExtValue();
        CurToken = Lx.getNextToken();
      }
      assert(CurToken == Token_Comma && "TODO: Error handle.");
      CurToken = Lx.getNextToken();
      if (CurToken == Token_Number) {
        UB = Lx.getAPInt().getSExtValue();
        CurToken = Lx.getNextToken();
      }
      assert(CurToken == Token_ParentheseClose && "TODO: Error handle.");
      CurToken = Lx.getNextToken();
    }
    CF->addSize(Name, UB, LB);
    if (CurToken == Token_Comma)
      CurToken = Lx.getNextToken();
    else
      break;
  }
}
void Parser::parseIndexDefines() {
  LLVM_DEBUG(dbgs() << "Parse index line\n");
  CurToken = Lx.getNextToken();
  while (CurToken == Token_Identifier) {
    auto Name = Lx.getIdentifier();
    int64_t Value = ConstIndexSymbol::Undefined;
    CurToken = Lx.getNextToken();
    if (CurToken == Token_Equal) {
      CurToken = Lx.getNextToken();
      assert(CurToken == Token_Number && "TODO: Error handle.");
      Value = Lx.getAPInt().getSExtValue();
      assert(Value >= 0 && "TODO: Error handle.");
      CurToken = Lx.getNextToken();
    }
    CF->addConstIndex(Name, Value);
    if (CurToken == Token_Comma)
      CurToken = Lx.getNextToken();
    else
      break;
  }
}

void Parser::parseInputDefines() {
  LLVM_DEBUG(dbgs() << "Parse input line\n");
  CurToken = Lx.getNextToken();
  while (CurToken == Token_Identifier) {
    auto Name = Lx.getIdentifier();
    CurToken = Lx.getNextToken();
    ShapeTy Shape;
    if (CurToken == Token_ParentheseOpen) {
      CurToken = Lx.getNextToken();
      while (true) {
        if (CurToken == Token_Identifier) {
          if (auto Size = CF->getSymbolOfName(Lx.getIdentifier())) {
            auto Sz0 = CF->addImplicitSize();
            Shape.emplace_back(Sz0);
            auto Sz1 = cast<SizeSymbol>(Size);
            SizeSymbol::makeSizeEqual(*Sz0, *Sz1);
          } else {
            Shape.emplace_back(CF->addSize(Name));
          }
          CurToken = Lx.getNextToken();
        } else if (CurToken == Token_Multiplication) {
          Shape.emplace_back(CF->addImplicitSize());
          CurToken = Lx.getNextToken();
        } else {
          break;
        }
        if (CurToken == Token_Comma)
          CurToken = Lx.getNextToken();
        else
          break;
      }
      assert(CurToken == Token_ParentheseClose);
      CurToken = Lx.getNextToken();
    }
    CF->addInputTensor(Name, Shape);
    if (CurToken == Token_Comma)
      CurToken = Lx.getNextToken();
    else
      break;
  }
}

void Parser::parseContractionTree() {
  // get Name
  auto Name = std::string(Lx.getIdentifier());
  CurToken = Lx.getNextToken();
  // parse Indexes
  IID = 0;
  NamesTy Names = parseIndexNames();
  auto Indexes = insertIndexDefines(Names, false);
  // parse expressions
  assert(CurToken == Token_Equal);
  CurToken = Lx.getNextToken();
  auto Child = parseAddition();
  eraseIndexDefines(Names);
  // create tree
  auto Root = RootNode::get(Indexes, std::move(Child));
  CF->addContractionTree(Name, std::move(Root));
}

NamesTy Parser::parseIndexNames() {
  NamesTy Names;
  if (CurToken != Token_ParentheseOpen)
    return Names;
  CurToken = Lx.getNextToken();
  while (CurToken == Token_Identifier) {
    Names.emplace_back(Lx.getIdentifier());
    CurToken = Lx.getNextToken();
    if (CurToken == Token_Comma)
      CurToken = Lx.getNextToken();
    else
      break;
  }
  assert(CurToken == Token_ParentheseClose && "TODO: Error handle.");
  CurToken = Lx.getNextToken();
  return Names;
}

NodePtr Parser::parseAddition() {
  ChildNodesTy Childs;
  while (true) {
    Childs.emplace_back(parseMultiplication());
    if (CurToken != Token_Addition)
      break;
    CurToken = Lx.getNextToken();
  }
  if (Childs.size() == 1)
    return std::move(Childs[0]);
  return AdditionNode::get(Childs);
}

NodePtr Parser::parseSingleNode() {
  if (CurToken == Token_Contraction) {
    return parseContraction();
  } else if (CurToken == Token_Negative) {
    return parseNegative();
  } else if (CurToken == Token_ParentheseOpen) {
    CurToken = Lx.getNextToken();
    auto Child = parseAddition();
    assert(CurToken == Token_ParentheseClose);
    CurToken = Lx.getNextToken();
    return Child;
  } else if (CurToken == Token_Identifier) {
    return parseIndexedTensor();
  } else {
    llvm_unreachable("TODO: Expression should begin with sum or identifier or "
                     "open parenthese or negative.");
  }
}

NodePtr Parser::parseNegative() {
  assert(CurToken == Token_Negative);
  CurToken = Lx.getNextToken();
  assert(CurToken != Token_Negative);
  return parseSingleNode();
}

NodePtr Parser::parseMultiplication() {
  ChildNodesTy Childs;
  while (true) {
    Childs.emplace_back(parseSingleNode());
    if (CurToken != Token_Multiplication)
      break;
    CurToken = Lx.getNextToken();
  }
  if (Childs.size() == 1)
    return std::move(Childs[0]);
  return MultiplicationNode::get(Childs);
}

NodePtr Parser::parseContraction() {
  assert(CurToken == Token_Contraction);
  CurToken = Lx.getNextToken();
  assert(CurToken == Token_ParentheseOpen);
  CurToken = Lx.getNextToken();
  // parse Indexes
  NamesTy Names = parseIndexNames();
  IndexesTy IndexDefs = insertIndexDefines(Names);
  CurToken = Lx.getNextToken();
  // parse expression
  auto child = parseAddition();
  assert(CurToken == Token_ParentheseClose);
  CurToken = Lx.getNextToken();
  eraseIndexDefines(Names);
  // create Node
  return ContractionNode::get(IndexDefs, std::move(child));
}

NodePtr Parser::parseIndexedTensor() {
  // get Name
  auto Name = std::string(Lx.getIdentifier());
  CurToken = Lx.getNextToken();
  // parse Indexes
  auto Names = parseIndexNames();
  auto IndexUses = getIndexUses(Names);
  auto T = CF->getSymbolOfName(Name);
  if (auto CT = llvm::dyn_cast_if_present<ContractionTree>(T)) {
    // parse indexed tensor
    inferenceIndexSizes(CT->getShape(), IndexUses);
    return TreeAsLeafNode::get(IndexUses, CT);
  } else {
    // parse Input tensor
    assert(!T || llvm::isa<InputTensor>(T));
    auto IT = llvm::cast_if_present<InputTensor>(T);
    if (!IT) {
      IT = CF->addInputTensor(Name, IndexUses.size());
    }
    inferenceIndexSizes(IT->getShape(), IndexUses);
    return InputAsLeafNode::get(IndexUses, IT);
  }
}

void Parser::parseKeepTensors() {
  assert(CurToken == Token_Keep);
  CurToken = Lx.getNextToken();
  while (CurToken == Token_Identifier) {
    auto Name = Lx.getIdentifier();
    CF->addKeep(Name);
    CurToken = Lx.getNextToken();
    if (CurToken != Token_Comma)
      break;
    CurToken = Lx.getNextToken();
  }
  assert(isNextLineToken());
  CurToken = Lx.getNextToken();
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
