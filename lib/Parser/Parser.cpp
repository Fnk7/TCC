#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "TCC/ContractionTree.h"
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

  Lexer &lexer;
  Token curToken;

  std::unique_ptr<ContractionForest> CF;

  // Indices
  IndexID indexID;
  DenseMap<StringRef, IndexPtr> indicesMap;

  inline bool isNextLineToken() {
    return curToken == Token_NewLine || curToken == Token_EOF;
  }

  // For statements
  void parseStatement();
  void parseSizeStatement();
  void parseInputStatement();
  void parseIndexStatement();
  void parseFormulaStatement();
  void parseKeepStatement();
  // For indices
  IndexUsesTy parseIndexUses();
  IndexDefsTy parseIndexDefines();
  void eraseIndexDefines(IndexDefsTy &indexDefs);
  // For arith
  NodePtr parseAdditionExpression();
  NodePtr parseContractionExpression();
  NodePtr parseMultiplicationExpression();
  NodePtr parseNegativeExpression();
  // For tensor indices
  NodePtr parseIndexedTensor();
  NodePtr parseExpressionFactor();

  template <typename Func> void parseCommaSplitTerms(Func parseTerm) {
    while (parseTerm() && curToken == Token_Comma) {
      curToken = lexer.getNextToken();
    }
  }

  void inferenceIndexSizes(SizesTy &shape, IndexUsesTy &indexUses) {
    assert(shape.size() == indexUses.size() && "TODO: Error handle.");
    for (auto [size, index] : llvm::zip(shape, indexUses)) {
      if (index->isImplicit())
        continue;
      if (index->getSize()) {
        SizeSymbol::makeSizeEqual(*size, *(index->getSize()));
      } else {
        index->setSize(size);
      }
    }
  }

public:
  Parser(Lexer &lexer) : lexer(lexer) {}
  std::unique_ptr<ContractionForest> parse();
};

std::unique_ptr<ContractionForest> Parser::parse() {
  CF = std::make_unique<ContractionForest>();
  curToken = lexer.getNextToken();
  while (curToken != Token_EOF) {
    parseStatement();
    curToken = lexer.getNextToken();
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
  curToken = lexer.getNextToken();
  auto parseTerm = [&]() -> bool {
    if (curToken != Token_Identifier)
      return false;
    auto name = lexer.getIdentifier();
    curToken = lexer.getNextToken();
    int64_t lowerBound = 0, upperBound = 0;
    if (curToken == Token_Equal) {
      curToken = lexer.getNextToken();
      if (curToken == Token_Number) {
        auto cst = lexer.getAPInt().getSExtValue();
        lowerBound = cst;
        upperBound = cst;
        curToken = lexer.getNextToken();
      } else {
        assert(curToken == Token_ParentheseOpen &&
               "Error: Size value/range define failed.");
        curToken = lexer.getNextToken();
        if (curToken == Token_Number) {
          lowerBound = lexer.getAPInt().getSExtValue();
          curToken = lexer.getNextToken();
        }
        assert(curToken == Token_Comma &&
               "Error: Size value/range define failed.");
        curToken = lexer.getNextToken();
        if (curToken == Token_Number) {
          upperBound = lexer.getAPInt().getSExtValue();
          curToken = lexer.getNextToken();
        }
        assert(curToken == Token_ParentheseClose &&
               "Error: Size value/range define failed.");
        curToken = lexer.getNextToken();
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
  curToken = lexer.getNextToken();
  auto parseTerm = [&]() -> bool {
    if (curToken != Token_Identifier)
      return false;
    auto name = lexer.getIdentifier();
    curToken = lexer.getNextToken();
    auto value = ParamIndexSymbol::Undefined;
    if (curToken == Token_Equal) {
      curToken = lexer.getNextToken();
      assert(curToken == Token_Number && "TODO: Error handle.");
      value = lexer.getAPInt().getSExtValue();
      assert(value >= 0 && "Error: Index defined should not less than 0.");
      curToken = lexer.getNextToken();
      CF->addParamIndexDefine(value, name);
    }
    return true;
  };
  parseCommaSplitTerms(parseTerm);
}

// Modified 23-03-28
void Parser::parseInputStatement() {
  LLVM_DEBUG(dbgs() << "Parsing input statement\n");
  curToken = lexer.getNextToken();
  SizesTy shape;
  auto parseSize = [&]() -> bool {
    if (curToken == Token_Identifier) {
      auto name = lexer.getIdentifier();
      if (auto size = CF->getSymbolIfPresent<SizeSymbol>(name)) {
        shape.emplace_back(size);
      } else {
        shape.emplace_back(CF->addSizeDefine(name));
      }
      curToken = lexer.getNextToken();
      return true;
    } else if (curToken == Token_Multiplication) {
      shape.emplace_back(CF->addSizeDefine());
      curToken = lexer.getNextToken();
      return true;
    }
    return false;
  };
  auto parseTerm = [&]() -> bool {
    if (curToken != Token_Identifier)
      return false;
    auto name = lexer.getIdentifier();
    curToken = lexer.getNextToken();
    shape.clear();
    if (curToken == Token_ParentheseOpen) {
      curToken = lexer.getNextToken();
      parseCommaSplitTerms(parseSize);
      assert(curToken == Token_ParentheseClose && "Error: Need token ')'.");
      curToken = lexer.getNextToken();
    }
    CF->addInputTensorDefine(name, shape);
    return true;
  };
  parseCommaSplitTerms(parseTerm);
}

void Parser::parseFormulaStatement() {
  LLVM_DEBUG(dbgs() << "Parsing formula statement\n");
  // get Name
  auto name = std::string(lexer.getIdentifier());
  curToken = lexer.getNextToken();
  // parse Indexes
  indexID = 0;
  IndexDefsTy indexDefs = parseIndexDefines();
  // parse expressions
  assert(curToken == Token_Equal);
  curToken = lexer.getNextToken();
  auto child = parseAdditionExpression();
  eraseIndexDefines(indexDefs);
  // create tree
  auto root = RootNode::get(indexDefs, std::move(child));
  CF->addContractionTree(name, std::move(root));
}

NodePtr Parser::parseAdditionExpression() {
  AdditionNode::SignsTy signs(1, true);
  ChildNodesTy childs;
  while (true) {
    childs.emplace_back(parseMultiplicationExpression());
    if (curToken == Token_Addition) {
      signs.emplace_back(true);
      curToken = lexer.getNextToken();
    } else if (curToken == Token_Negative) {
      signs.emplace_back(false);
      curToken = lexer.getNextToken();
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

NodePtr Parser::parseMultiplicationExpression() {
  ChildNodesTy childs;
  while (true) {
    childs.emplace_back(parseExpressionFactor());
    if (curToken != Token_Multiplication)
      break;
    curToken = lexer.getNextToken();
  }
  if (childs.size() == 1)
    return std::move(childs[0]);
  return MultiplicationNode::get(childs);
}

NodePtr Parser::parseExpressionFactor() {
  if (curToken == Token_Identifier) {
    return parseIndexedTensor();
  } else if (curToken == Token_Contraction) {
    return parseContractionExpression();
  } else if (curToken == Token_ParentheseOpen) {
    curToken = lexer.getNextToken();
    auto child = parseAdditionExpression();
    assert(curToken == Token_ParentheseClose);
    curToken = lexer.getNextToken();
    return child;
  } else if (curToken == Token_Negative) {
    return parseNegativeExpression();
  } else if (curToken == Token_Number) {
    auto value = lexer.getAPFloat().convertToDouble();
    curToken = lexer.getNextToken();
    return ConstantNode::get(value);
  } else {
    llvm_unreachable("Error: Expression should begin with identifier, sum, "
                     "number, '(' or '-'.");
  }
}

// Modified 23-03-28
IndexDefsTy Parser::parseIndexDefines() {
  IndexDefsTy indexDefs;
  if (curToken != Token_ParentheseOpen)
    return indexDefs;
  curToken = lexer.getNextToken();
  auto parseIndexDef = [&]() -> bool {
    if (curToken != Token_Identifier)
      return false;
    auto name = lexer.getIdentifier();
    curToken = lexer.getNextToken();
    assert(!indicesMap.count(name) &&
           "Error: Index definition already exists.");
    auto *index = new IterIndexSymbol(indexID++, name);
    indicesMap[index->getName()] = index;
    indexDefs.emplace_back(index);
    return true;
  };
  parseCommaSplitTerms(parseIndexDef);
  assert(curToken == Token_ParentheseClose && "TODO: Error handle.");
  curToken = lexer.getNextToken();
  return indexDefs;
}
void Parser::eraseIndexDefines(IndexDefsTy &indexDefs) {
  for (auto &index : indexDefs) {
    assert(indicesMap.erase(index->getName()));
  }
}

NodePtr Parser::parseNegativeExpression() {
  assert(curToken == Token_Negative);
  curToken = lexer.getNextToken();
  assert(curToken != Token_Negative);
  auto child = parseExpressionFactor();
  return NegativeNode::get(std::move(child));
}

// Done 12-08
NodePtr Parser::parseContractionExpression() {
  assert(curToken == Token_Contraction);
  curToken = lexer.getNextToken();
  assert(curToken == Token_ParentheseOpen);
  curToken = lexer.getNextToken();
  // parse Indexes
  IndexDefsTy indexDefs = parseIndexDefines();
  curToken = lexer.getNextToken();
  // parse expression
  auto child = parseAdditionExpression();
  assert(curToken == Token_ParentheseClose);
  curToken = lexer.getNextToken();
  eraseIndexDefines(indexDefs);
  // create Node
  return ContractionNode::get(indexDefs, std::move(child));
}

// Done 12-08
IndexUsesTy Parser::parseIndexUses() {
  IndexUsesTy indexUses;
  if (curToken != Token_ParentheseOpen)
    return indexUses;
  curToken = lexer.getNextToken();
  auto parseIndexUse = [&]() -> bool {
    if (curToken == Token_Identifier) {
      curToken = lexer.getNextToken();
      // Iter or Param index
      auto name = lexer.getIdentifier();
      if (indicesMap.count(name))
        indexUses.emplace_back(indicesMap[name]);
      else
        indexUses.emplace_back(CF->getParamIndex(name));
      return true;
    } else if (curToken == Token_Number) {
      curToken = lexer.getNextToken();
      // Implicit constant index
      auto value = lexer.getAPInt().getSExtValue();
      assert(value >= 0);
      indexUses.emplace_back(CF->getParamIndex(value));
      return true;
    }
    return false;
  };
  parseCommaSplitTerms(parseIndexUse);
  assert(curToken == Token_ParentheseClose && "TODO: Error handle.");
  curToken = lexer.getNextToken();
  return indexUses;
}

// Done 12-08
NodePtr Parser::parseIndexedTensor() {
  // Parse tensor
  auto name = std::string(lexer.getIdentifier());
  curToken = lexer.getNextToken();
  // Parse indices
  auto indexUses = parseIndexUses();
  auto tensor = CF->getSymbol(name);
  if (auto treeTensor = llvm::dyn_cast_if_present<ContractionTree>(tensor)) {
    // Parse contracted tensor
    inferenceIndexSizes(treeTensor->getShape(), indexUses);
    return TreeTensorNode::get(indexUses, treeTensor);
  } else {
    // Parse input tensor
    assert(!tensor || llvm::isa<InputTensor>(tensor));
    auto inputTensor = llvm::cast_if_present<InputTensor>(tensor);
    if (!inputTensor) {
      inputTensor = CF->addInputTensorDefine(name, indexUses.size());
    }
    inferenceIndexSizes(inputTensor->getShape(), indexUses);
    return InputTensorNode::get(indexUses, inputTensor);
  }
}

// Modified 23-03-28
void Parser::parseKeepStatement() {
  LLVM_DEBUG(dbgs() << "Parsing keep statement\n");
  curToken = lexer.getNextToken();
  auto parseTerm = [&]() -> bool {
    if (curToken != Token_Identifier)
      return false;
    auto name = lexer.getIdentifier();
    CF->addTensorKeeped(name);
    curToken = lexer.getNextToken();
    return true;
  };
  parseCommaSplitTerms(parseTerm);
}
} // namespace parser

std::unique_ptr<ContractionForest> parseInputFile(StringRef filename) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> fileOrErr =
      MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code EC = fileOrErr.getError()) {
    errs() << "Open input file failed: " << EC.message() << "\n";
    return nullptr;
  }
  parser::Lexer lexer{std::move(fileOrErr.get())};
  parser::Parser parser{lexer};
  auto CF = parser.parse();
  return CF;
}
} // namespace tc

#undef DEBUG_TYPE
