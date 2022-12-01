#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "TCC/Parser.h"

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
};

class Lexer {
  std::unique_ptr<llvm::MemoryBuffer> MemoryBuffer;
  StringRef Buffer;
  int LastChar;
  size_t Position;
  std::string Identifier;

  inline int getNextChar() {
    if (Position >= Buffer.size())
      return EOF;
    return Buffer[Position++];
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
      Identifier = "";
      do {
        Identifier += (char)LastChar;
        LastChar = getNextChar();
      } while (isalnum(LastChar) || LastChar == '_');
      if (Identifier == "sum")
        return Token_Contraction;
      if (Identifier == "keep")
        return Token_Keep;
      return Token_Identifier;
    }
    Token T{LastChar};
    LastChar = getNextChar();
    return T;
  }
  std::string getIdentifier() { return Identifier; }
};

class Parser {
  using NamedIndexesTy = SmallVector<std::string, 4>;

  Lexer &Lx;
  Token CurToken;

  SmallVector<SizePtr> IndexSizeMap;
  DenseMap<StringRef, int> IndexMap;

  std::unique_ptr<ContractionForest> CF;

  inline bool isNextLineToken() {
    return CurToken == Token_NewLine || CurToken == Token_EOF;
  }

  void parseLine();
  void parseKeepTensors();
  void parseContractionTree();
  NamedIndexesTy parseNamedIndexes();
  NodePtr parseAddition();
  NodePtr parseMultiplication();
  NodePtr parseIndexedTensor();
  NodePtr parseContraction();

  IndexesTy insertIndexesMap(NamedIndexesTy &NamedIndexes) {
    IndexesTy Indexes;
    for (auto &Name : NamedIndexes) {
      assert(!IndexMap.count(Name));
      IndexMap[Name] = IndexSizeMap.size();
      Indexes.emplace_back(IndexSizeMap.size());
      IndexSizeMap.emplace_back(nullptr);
    }
    return Indexes;
  }
  void eraseIndexesMap(NamedIndexesTy &NamedIndexes) {
    for (auto &Name : NamedIndexes) {
      assert(IndexMap.count(Name));
      IndexMap.erase(Name);
    }
  }
  IndexesTy getIndexes(NamedIndexesTy &NamedIndexes) {
    IndexesTy Indexes;
    for (auto &Name : NamedIndexes) {
      assert(IndexMap.count(Name));
      Indexes.emplace_back(IndexMap[Name]);
    }
    return Indexes;
  }
  SizesTy getSizes(IndexesTy &Indexes) {
    SizesTy Sizes;
    for (auto index : Indexes) {
      Sizes.emplace_back(IndexSizeMap[index]);
    }
    return Sizes;
  }

  void setSizes(IndexesTy &Indexes, SizesTy &Sizes) {
    assert(Indexes.size() == Sizes.size());
    for (size_t i = 0; i < Indexes.size(); i++) {
      IndexSizeMap[Indexes[i]] = Sizes[i];
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
  else
    assert("Line should begin with keep or tensor identifier.");
  assert(isNextLineToken());
}

void Parser::parseContractionTree() {
  // get Name
  auto Name = std::string(Lx.getIdentifier());
  CurToken = Lx.getNextToken();
  // parse Indexes
  IndexSizeMap.clear();
  NamedIndexesTy NamedIndexes = parseNamedIndexes();
  auto Indexes = insertIndexesMap(NamedIndexes);
  // parse expressions
  assert(CurToken == Token_Equal);
  CurToken = Lx.getNextToken();
  auto Child = parseAddition();
  eraseIndexesMap(NamedIndexes);
  // create tree
  auto Sizes = getSizes(Indexes);
  auto Root = RootNode::get(Sizes, Indexes, std::move(Child));
  CF->addContractionTree(Name, std::move(Root));
}

Parser::NamedIndexesTy Parser::parseNamedIndexes() {
  NamedIndexesTy NamedIndexes;
  if (CurToken != Token_ParentheseOpen)
    return NamedIndexes;
  CurToken = Lx.getNextToken();
  while (CurToken == Token_Identifier) {
    NamedIndexes.emplace_back(std::string(Lx.getIdentifier()));
    CurToken = Lx.getNextToken();
    if (CurToken == Token_Comma)
      CurToken = Lx.getNextToken();
    else
      break;
  }
  assert(CurToken == Token_ParentheseClose);
  CurToken = Lx.getNextToken();
  return NamedIndexes;
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

NodePtr Parser::parseMultiplication() {
  ChildNodesTy Childs;
  while (true) {
    if (CurToken == Token_Contraction)
      Childs.emplace_back(parseContraction());
    else if (CurToken == Token_Identifier)
      Childs.emplace_back(parseIndexedTensor());
    else if (CurToken == Token_ParentheseOpen) {
      CurToken = Lx.getNextToken();
      Childs.emplace_back(parseAddition());
      assert(CurToken == Token_ParentheseClose);
      CurToken = Lx.getNextToken();
    } else
      assert(
          "Expression should begin with sum or identifier or open parenthese.");
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
  NamedIndexesTy NamedIndexes = parseNamedIndexes();
  IndexesTy Indexes = insertIndexesMap(NamedIndexes);
  CurToken = Lx.getNextToken();
  // parse expression
  auto child = parseAddition();
  assert(CurToken == Token_ParentheseClose);
  CurToken = Lx.getNextToken();
  eraseIndexesMap(NamedIndexes);
  // create Node
  auto Sizes = getSizes(Indexes);
  return ContractionNode::get(Sizes, Indexes, std::move(child));
}

NodePtr Parser::parseIndexedTensor() {
  // get Name
  auto Name = std::string(Lx.getIdentifier());
  CurToken = Lx.getNextToken();
  // parse Indexes
  NamedIndexesTy NamedIndexes = parseNamedIndexes();
  IndexesTy Indexes = getIndexes(NamedIndexes);
  ShapeTy Sz1 = getSizes(Indexes);
  auto T = CF->getTensorOfName(Name);
  if (auto CT = llvm::dyn_cast_if_present<ContractionTree>(T)) {
    // parse indexed tensor
    CF->setSizeEqual(Sz1, CT->getShape());
    setSizes(Indexes, Sz1);
    return TreeAsLeafNode::get(Indexes, CT);
  } else {
    // parse Input tensor
    assert(!T || llvm::isa<InputTensor>(T));
    auto IT = llvm::cast_if_present<InputTensor>(T);
    if (!IT) {
      IT = CF->addInputTensor(Name, Indexes.size());
    }
    CF->setSizeEqual(Sz1, IT->getShape());
    setSizes(Indexes, Sz1);
    return InputAsLeafNode::get(Indexes, IT);
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
