#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "TCC/Parser.h"

#define DEBUG_TYPE "tcc-parser"

using namespace llvm;

namespace tc {
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
private:
  std::unique_ptr<llvm::MemoryBuffer> memoryBuffer;
  StringRef buffer;
  int lastChar = ' ';
  size_t position = 0;
  std::string termString;

  DenseMap<StringRef, Token> keywords;

  inline int getNextChar() {
    if (position >= buffer.size())
      return EOF;
    return buffer[position++];
  }

  void setNumberStr(bool positive = true) {
    termString = positive ? "" : "-";
    do {
      termString += (char)lastChar;
      lastChar = getNextChar();
    } while (isdigit(lastChar) || lastChar == '.');
  }

public:
  Lexer(std::unique_ptr<llvm::MemoryBuffer> memoryBuffer)
      : memoryBuffer(std::move(memoryBuffer)) {
    buffer = this->memoryBuffer->getBuffer();
    keywords["index"] = Token_Index;
    keywords["size"] = Token_Size;
    keywords["input"] = Token_Input;
    keywords["sum"] = Token_Contraction;
    keywords["keep"] = Token_Keep;
  }

  Token getNextToken() {
    while (lastChar != '\n' && isspace(lastChar))
      lastChar = getNextChar();
    if (isalpha(lastChar)) {
      termString.clear();
      do {
        termString += (char)lastChar;
        lastChar = getNextChar();
      } while (isalnum(lastChar) || lastChar == '_');
      if (keywords.count(termString))
        return keywords[termString];
      return Token_Identifier;
    }
    if (lastChar == '-') {
      lastChar = getNextChar();
      if (isdigit(lastChar)) {
        setNumberStr(false);
        return Token_Number;
      } else {
        return Token_Negative;
      }
    }
    if (isdigit(lastChar)) {
      setNumberStr();
      return Token_Number;
    }
    Token T{lastChar};
    lastChar = getNextChar();
    return T;
  }

  std::string getIdentifier() { return termString; }
  APInt getAPInt() { return APInt(64, termString, 10); }
  APFloat getAPFloat() { return APFloat(APFloat::IEEEdouble(), termString); }
};

using NamesTy = SmallVector<std::string, 4>;
class Parser {

  Lexer &lx;
  Token curToken;

  std::unique_ptr<ContractionForest> CF;

  // Indices
  IndexID indexID;
  DenseMap<StringRef, IndexPtr> indicesMap;

  inline bool isNextLineToken() {
    return curToken == Token_NewLine || curToken == Token_EOF;
  }

  void parseStatement();
  void parseSizeStatement();
  void parseInputStatement();
  void parseIndexStatement();
  void parseFormulaStatement();
  void parseKeepStatement();
  IndexUsesTy parseIndexUses();
  IndexDefsTy parseIndexDefines();
  void eraseIterIndexDefines(IndexDefsTy &IndexDefs);
  NamesTy parseIndexNames();
  NodePtr parseAdditionExpression();
  NodePtr parseSingleNode();
  NodePtr parseNegative();
  NodePtr parseMultiplicationExpression();
  NodePtr parseIndexedTensor();
  NodePtr parseContraction();

  template <typename Func> void parseCommaSplitTerms(Func parseTerm) {
    while (parseTerm() && curToken == Token_Comma) {
      curToken = lx.getNextToken();
    }
  }

  void inferenceIndexSizes(SizesTy &Shape, IndexUsesTy &IndexUses) {
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
  Parser(Lexer &lx) : lx(lx) {}
  std::unique_ptr<ContractionForest> parse();
};

std::unique_ptr<ContractionForest> Parser::parse() {
  CF = std::make_unique<ContractionForest>();
  curToken = lx.getNextToken();
  while (curToken != Token_EOF) {
    parseStatement();
    curToken = lx.getNextToken();
  }
  // TODO: After Parse
  CF->finishForestConstruct();
  return std::move(CF);
}

void Parser::parseStatement() {
  if (curToken == Token_Index)
    parseIndexStatement();
  else if (curToken == Token_Size)
    parseSizeStatement();
  else if (curToken == Token_Input)
    parseInputStatement();
  else if (curToken == Token_Identifier)
    parseFormulaStatement();
  else if (curToken == Token_Keep)
    parseKeepStatement();
  else if (!isNextLineToken())
    llvm_unreachable("TODO: Error handle.");
  assert(isNextLineToken());
}

// Modified 23-03-28
void Parser::parseSizeStatement() {
  LLVM_DEBUG(dbgs() << "Parsing size statement\n");
  curToken = lx.getNextToken();
  auto parseTerm = [&]() -> bool {
    if (curToken != Token_Identifier)
      return false;
    auto name = lx.getIdentifier();
    curToken = lx.getNextToken();
    int64_t lowerBound = 0, upperBound = 0;
    if (curToken == Token_Equal) {
      curToken = lx.getNextToken();
      if (curToken == Token_Number) {
        auto cst = lx.getAPInt().getSExtValue();
        lowerBound = cst;
        upperBound = cst;
        curToken = lx.getNextToken();
      } else {
        assert(curToken == Token_ParentheseOpen &&
               "Error: Size value/range define failed.");
        curToken = lx.getNextToken();
        if (curToken == Token_Number) {
          lowerBound = lx.getAPInt().getSExtValue();
          curToken = lx.getNextToken();
        }
        assert(curToken == Token_Comma &&
               "Error: Size value/range define failed.");
        curToken = lx.getNextToken();
        if (curToken == Token_Number) {
          upperBound = lx.getAPInt().getSExtValue();
          curToken = lx.getNextToken();
        }
        assert(curToken == Token_ParentheseClose &&
               "Error: Size value/range define failed.");
        curToken = lx.getNextToken();
      }
    }
    CF->addSizeDefine(name, upperBound, lowerBound);
    return true;
  };
  parseCommaSplitTerms(parseTerm);
}

// Modified 23-03-28
void Parser::parseIndexStatement() {
  LLVM_DEBUG(dbgs() << "Parsing index statement\n");
  curToken = lx.getNextToken();
  auto parseTerm = [&]() -> bool {
    if (curToken != Token_Identifier)
      return false;
    auto name = lx.getIdentifier();
    curToken = lx.getNextToken();
    auto value = ParamIndexSymbol::Undefined;
    if (curToken == Token_Equal) {
      curToken = lx.getNextToken();
      assert(curToken == Token_Number && "TODO: Error handle.");
      value = lx.getAPInt().getSExtValue();
      assert(value >= 0 && "Error: Index defined should not less than 0.");
      curToken = lx.getNextToken();
      CF->addParamIndexDefine(value, name);
    }
    return true;
  };
  parseCommaSplitTerms(parseTerm);
}

// Modified 23-03-28
void Parser::parseInputStatement() {
  LLVM_DEBUG(dbgs() << "Parsing input statement\n");
  curToken = lx.getNextToken();
  SizesTy shape;
  auto parseSize = [&]() -> bool {
    if (curToken == Token_Identifier) {
      auto name = lx.getIdentifier();
      if (auto symbol = CF->getSymbolOfName(name)) {
        auto size = cast<SizeSymbol>(symbol);
        shape.emplace_back(size);
      } else {
        shape.emplace_back(CF->addSizeDefine(name));
      }
      curToken = lx.getNextToken();
      return true;
    } else if (curToken == Token_Multiplication) {
      shape.emplace_back(CF->addSizeDefine());
      curToken = lx.getNextToken();
      return true;
    }
    return false;
  };
  auto parseTerm = [&]() -> bool {
    if (curToken != Token_Identifier)
      return false;
    auto name = lx.getIdentifier();
    curToken = lx.getNextToken();
    shape.clear();
    if (curToken == Token_ParentheseOpen) {
      curToken = lx.getNextToken();
      parseCommaSplitTerms(parseSize);
      assert(curToken == Token_ParentheseClose && "Error: Need token ')'.");
      curToken = lx.getNextToken();
    }
    CF->addInputTensorDefine(name, shape);
    return true;
  };
  parseCommaSplitTerms(parseTerm);
}

void Parser::parseFormulaStatement() {
  // get Name
  auto name = std::string(lx.getIdentifier());
  curToken = lx.getNextToken();
  // parse Indexes
  indexID = 0;
  IndexDefsTy indexDefs = parseIndexDefines();
  // parse expressions
  assert(curToken == Token_Equal);
  curToken = lx.getNextToken();
  auto child = parseAdditionExpression();
  eraseIterIndexDefines(indexDefs);
  // create tree
  auto root = RootNode::get(indexDefs, std::move(child));
  CF->addContractionTree(name, std::move(root));
}

NamesTy Parser::parseIndexNames() {
  NamesTy names;
  if (curToken != Token_ParentheseOpen)
    return names;
  curToken = lx.getNextToken();
  auto parseTerm = [&]() -> bool {
    while (curToken != Token_Identifier)
      return false;
    names.emplace_back(lx.getIdentifier());
    curToken = lx.getNextToken();
    return true;
  };
  parseCommaSplitTerms(parseTerm);
  assert(curToken == Token_ParentheseClose && "TODO: Error handle.");
  curToken = lx.getNextToken();
  return names;
}

NodePtr Parser::parseAdditionExpression() {
  AdditionNode::SignsTy signs(1, true);
  ChildNodesTy childs;
  while (true) {
    childs.emplace_back(parseMultiplicationExpression());
    if (curToken == Token_Addition) {
      signs.emplace_back(true);
      curToken = lx.getNextToken();
    } else if (curToken == Token_Negative) {
      signs.emplace_back(false);
      curToken = lx.getNextToken();
      assert(curToken != Token_Negative &&
             "TODO: double negative, too complex.");
    } else {
      break;
    }
  }
  if (childs.size() == 1)
    return std::move(childs[0]);
  return AdditionNode::get(childs, signs);
}

NodePtr Parser::parseSingleNode() {
  if (curToken == Token_Contraction) {
    return parseContraction();
  } else if (curToken == Token_Negative) {
    return parseNegative();
  } else if (curToken == Token_ParentheseOpen) {
    curToken = lx.getNextToken();
    auto Child = parseAdditionExpression();
    assert(curToken == Token_ParentheseClose);
    curToken = lx.getNextToken();
    return Child;
  } else if (curToken == Token_Identifier) {
    return parseIndexedTensor();
  } else if (curToken == Token_Number) {
    auto Value = lx.getAPFloat().convertToDouble();
    curToken = lx.getNextToken();
    return ConstantNode::get(Value);
  } else {
    llvm_unreachable("TODO: Expression should begin with sum or identifier or "
                     "open parenthese or negative.");
  }
}

NodePtr Parser::parseNegative() {
  assert(curToken == Token_Negative);
  curToken = lx.getNextToken();
  assert(curToken != Token_Negative);
  auto Child = parseSingleNode();
  return NegativeNode::get(std::move(Child));
}

NodePtr Parser::parseMultiplicationExpression() {
  ChildNodesTy Childs;
  while (true) {
    Childs.emplace_back(parseSingleNode());
    if (curToken != Token_Multiplication)
      break;
    curToken = lx.getNextToken();
  }
  if (Childs.size() == 1)
    return std::move(Childs[0]);
  return MultiplicationNode::get(Childs);
}

// Done 12-08
IndexDefsTy Parser::parseIndexDefines() {
  NamesTy names = parseIndexNames();
  IndexDefsTy indexDefs;
  for (auto name : names) {
    assert(!indicesMap.count(name) &&
           "Error: Index definition already exists.");
    auto *index = new IterIndexSymbol(indexID++, name);
    indicesMap[index->getName()] = index;
    indexDefs.emplace_back(index);
  }
  return indexDefs;
}
void Parser::eraseIterIndexDefines(IndexDefsTy &indexDefs) {
  for (auto &index : indexDefs) {
    assert(indicesMap.erase(index->getName()));
  }
}

// Done 12-08
NodePtr Parser::parseContraction() {
  assert(curToken == Token_Contraction);
  curToken = lx.getNextToken();
  assert(curToken == Token_ParentheseOpen);
  curToken = lx.getNextToken();
  // parse Indexes
  IndexDefsTy IndexDefs = parseIndexDefines();
  curToken = lx.getNextToken();
  // parse expression
  auto child = parseAdditionExpression();
  assert(curToken == Token_ParentheseClose);
  curToken = lx.getNextToken();
  eraseIterIndexDefines(IndexDefs);
  // create Node
  return ContractionNode::get(IndexDefs, std::move(child));
}

// Done 12-08
IndexUsesTy Parser::parseIndexUses() {
  IndexUsesTy IndexUses;
  if (curToken != Token_ParentheseOpen)
    return IndexUses;
  curToken = lx.getNextToken();
  while (true) {
    if (curToken == Token_Identifier) {
      // Iter index
      auto Name = lx.getIdentifier();
      if (indicesMap.count(Name)) {
        IndexUses.emplace_back(indicesMap[Name]);
      } else {
        // Param Index
        auto PIndex = CF->getParamIndex(Name);
        IndexUses.emplace_back(PIndex);
      }
    } else if (curToken == Token_Number) {
      // Const index
      auto Value = lx.getAPInt().getSExtValue();
      assert(Value >= 0);
      auto PIndex = CF->getParamIndex(Value);
      IndexUses.emplace_back(PIndex);
    } else
      break;
    curToken = lx.getNextToken();
    if (curToken == Token_Comma)
      curToken = lx.getNextToken();
    else
      break;
  }
  assert(curToken == Token_ParentheseClose && "TODO: Error handle.");
  curToken = lx.getNextToken();
  return IndexUses;
}

// Done 12-08
NodePtr Parser::parseIndexedTensor() {
  // Parse tensor
  auto Name = std::string(lx.getIdentifier());
  curToken = lx.getNextToken();
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

// Modified 23-03-28
void Parser::parseKeepStatement() {
  LLVM_DEBUG(dbgs() << "Parsing keep statement\n");
  curToken = lx.getNextToken();
  auto parseTerm = [&]() -> bool {
    if (curToken != Token_Identifier)
      return false;
    auto name = lx.getIdentifier();
    CF->addTensorKeeped(name);
    curToken = lx.getNextToken();
    return true;
  };
  parseCommaSplitTerms(parseTerm);
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
  auto CF = Parser.parse();
  return CF;
}
} // namespace tc

#undef DEBUG_TYPE
