#include "TCC/CodeGen.h"

#include "TCC/Dialect/TBLAS/IR/TBLASOps.h"

#include "TCC/Dialect/TC/IR/TCOps.h"

#include "TCC/Dialect/TC/Utils/Utils.h"
#include "TCC/Dialect/Utils/FuncUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::tc;
using namespace mlir::tc::utils;

#define DEBUG_TYPE "tc-linalg-c"

namespace {
class EmitCImpl : public codegen::CodeGenBase {
  func::FuncOp funcOp;
  Block &block;
  ReturnOp returnOp;

public:
  EmitCImpl(func::FuncOp funcOp, llvm::raw_ostream &out)
      : CodeGenBase(out), funcOp(funcOp), block(funcOp.front()) {
    returnOp = dyn_cast<ReturnOp>(block.getTerminator());
  }

  void emit() {
    emitFunctionType();
    emitFunctionBody();
  }

private:
  int64_t indent = 0;

  using ValueNameMapTy = DenseMap<Value, std::string>;
  ValueNameMapTy valueNameMap;

  // emit size
  using SizeIDSymbolMapTy = DenseMap<int64_t, std::string>;
  SizeIDSymbolMapTy sizeIDNameMap;
  void emitConstantOrVaribaleSize(int64_t size, int64_t sizeID) {
    if (size >= 0) {
      out << size;
    } else {
      out << sizeIDNameMap[sizeID];
    }
  }
  // emit local array
  int64_t localIdx = 0;
  void emitLocalArrayForTensor(Value value) {
    if (valueNameMap.count(value)) {
      return;
    }
    std::string name = "local_" + std::to_string(localIdx++);
    out.indent(indent) << "double " << name;
    auto [shape, sizeIDs] = getTensorSizeInfos(value);
    for (auto [size, sizeID] : llvm::zip(shape, sizeIDs)) {
      out << "[";
      emitConstantOrVaribaleSize(size, sizeID);
      out << "]";
    }
    out << ";\n";
    valueNameMap[value] = name;
  }

private:
  bool isValidName(StringRef name) {
    for (auto c : name)
      if (!isalnum(c) && !(c == '_'))
        return false;
    return true;
  }
  std::string getArgName(int64_t argIdx) {
    auto name = getFuncOpArgName(funcOp, argIdx);
    if (isValidName(name))
      return std::string(name);
    return "arg" + std::to_string(argIdx);
  }
  std::string getResultName(int64_t resultIdx) {
    auto name = getFuncOpResultName(funcOp, resultIdx);
    if (isValidName(name))
      return std::string(name);
    return "ret" + std::to_string(resultIdx);
  }

  void emitFunctionType() {
    std::string funcDeclare = "void ";
    out << "void ";
    indent = 5;
    out << funcOp.getName();
    indent += funcOp.getName().size();
    out << "(";
    indent += 1;
    bool newLine = false;
    emitArgumentTypes(newLine);
    emitResultTypes(newLine);
    out << ")\n";
  }

  void emitArgumentTypes(bool &newLine) {
    auto inputs = block.getArguments();
    for (auto [idx, input] : llvm::enumerate(inputs)) {
      if (newLine) {
        out << ",\n";
        out.indent(indent);
      }
      newLine = true;
      auto name = getArgName(idx);
      if (input.getType().isIndex()) {
        emitIndexArgument(input, idx, name);
      } else {
        emitTensorArgument(input, name);
      }
      valueNameMap[input] = name;
    }
  }

  void emitIndexArgument(Value value, int64_t sizeID, std::string name) {
    out << "int " << name;
    sizeIDNameMap[sizeID] = name;
  }

  void emitTensorArgument(Value value, std::string name) {
    out << "double " << name;
    auto [shape, sizeIDs] = getTensorSizeInfos(value);
    for (auto [size, sizeID] : llvm::zip(shape, sizeIDs)) {
      out << "[";
      emitConstantOrVaribaleSize(size, sizeID);
      out << "]";
    }
  }

  SmallVector<std::string> returnNames;
  void emitResultTypes(bool &newLine) {
    // TODO: use writebacks
    for (auto [idx, input] : llvm::enumerate(returnOp.getOperands())) {
      if (newLine) {
        out << ",\n";
        out.indent(indent);
      }
      newLine = true;
      auto name = getResultName(idx);
      emitTensorArgument(input, name);
      valueNameMap[input] = name;
      returnNames.emplace_back(name);
    }
  }

  void emitFunctionBody() {
    out << "{\n";
    out << "#pragma scop\n";
    indent = 2;
    for (auto &op : block) {
      if (auto cstOp = dyn_cast_or_null<arith::ConstantFloatOp>(&op)) {
        emitArithConstantOp(cstOp);
      } else if (auto linalgOp = dyn_cast_or_null<linalg::GenericOp>(&op)) {
        emitLinalgGenericOp(linalgOp);
      } else if (auto fillOp = dyn_cast_or_null<linalg::FillOp>(&op)) {
        emitLinalgFillOp(fillOp);
      } else if (auto blasOp = dyn_cast_or_null<GenericOp>(&op)) {
        out.indent(indent) << "// Failed: Emit C for op: " << op.getName()
                           << "\n";
      } else if (auto emptyOp = dyn_cast_or_null<tensor::EmptyOp>(&op)) {
        emitTensorEmptyOp(emptyOp);
      } else if (auto returnOp = dyn_cast_or_null<ReturnOp>(&op)) {
        emitFuncReturnOp(returnOp);
      } else {
        out.indent(indent) << "// Failed: Emit C for op: " << op.getName()
                           << "\n";
      }
    }
    out << "#pragma endscop\n";
    out << "}\n";
  }

  // emit arith constant
  int64_t cstIdx = 0;
  void emitArithConstantOp(arith::ConstantFloatOp op) {
    std::string name = "cst" + std::to_string(cstIdx++);
    SmallVector<char> value;
    op.value().toString(value);
    out.indent(indent) << "double " << name << " = " << value << ";\n";
    valueNameMap[op.getResult()] = name;
  }

  // emit tensor empty op
  void emitTensorEmptyOp(tensor::EmptyOp op) {
    out.indent(indent) << "// From Tensor EmptyOp\n";
    emitLocalArrayForTensor(op.getResult());
  }

  // emit func return op
  void emitFuncReturnOp(ReturnOp op) {
    out.indent(indent) << "// From ReturnOp\n";
    auto outputs = op.getOperands();
    for (auto [idx, output] : llvm::enumerate(outputs)) {
      auto [shape, sizeIDs] = getTensorSizeInfos(output);
      int numLoops = shape.size();
      SmallVector<std::string> iterVars(numLoops, "");
      for (auto [idx, size] : llvm::enumerate(shape)) {
        iterVars[idx] = getIteratorName(idx);
        emitForLoop(iterVars[idx], size, sizeIDs[idx]);
      }
      out.indent(indent) << returnNames[idx];
      emitIterVars(iterVars);
      out << " = " << valueNameMap[output];
      emitIterVars(iterVars);
      out << ";\n";
      emitForLoopsEnd(numLoops);
    }
  }

  // emit linalg fill op
  void emitLinalgFillOp(linalg::FillOp op) {
    out.indent(indent) << "// From Linalg FillOp\n";
    assert(op.getResults().size() == 1);
    auto input = op.getInputs()[0];
    auto output = op.getOutputs()[0];
    auto result = op.getResults()[0];
    valueNameMap[result] = valueNameMap[output];
    auto [shape, sizeIDs] = getTensorSizeInfos(output);
    int numLoops = shape.size();
    SmallVector<std::string> iterVars(numLoops, "");
    for (auto [idx, size] : llvm::enumerate(shape)) {
      iterVars[idx] = getIteratorName(idx);
      emitForLoop(iterVars[idx], size, sizeIDs[idx]);
    }
    out.indent(indent) << valueNameMap[result];
    emitIterVars(iterVars);
    out << " = " << valueNameMap[input] << ";\n";
    emitForLoopsEnd(numLoops);
  }

  // emit linalg generic op
  using PositionIterVariableMapTy = DenseMap<unsigned, std::string>;
  void emitLinalgGenericOp(linalg::GenericOp op) {
    out.indent(indent) << "// From Linalg GenericOp\n";
    auto operands = op->getOperands();
    auto inputs = op.getInputs();
    auto outputs = op.getOutputs();
    auto results = op.getResults();
    auto indexingMaps = op.getIndexingMapsArray();
    auto iteratorTypes = op.getIteratorTypesArray();
    // 1. emit parallel loop
    int numLoops = iteratorTypes.size();
    SmallVector<std::string> iterVars(numLoops, "");
    int numParallelLoops = 0;
    for (auto [idx, output] : llvm::enumerate(outputs)) {
      auto affineMap = indexingMaps[inputs.size() + idx];
      auto exprs = affineMap.getResults();
      auto [shape, sizeIDs] = getTensorSizeInfos(output);
      for (auto [expr, size, sizeID] : llvm::zip(exprs, shape, sizeIDs)) {
        auto position = expr.cast<AffineDimExpr>().getPosition();
        if (iterVars[position] != "") {
          continue;
        }
        assert(iteratorTypes[position] == "parallel");
        numParallelLoops++;
        iterVars[position] = getIteratorName(position);
        emitForLoop(iterVars[position], size, sizeID);
      }
    }
    // 2. emit reduction loop
    int numReductionLoops = 0;
    if (numParallelLoops < numLoops) {
      for (auto [idx, input] : llvm::enumerate(inputs)) {
        auto affineMap = indexingMaps[idx];
        auto exprs = affineMap.getResults();
        auto [shape, sizeIDs] = getTensorSizeInfos(input);
        for (auto [expr, size, sizeID] : llvm::zip(exprs, shape, sizeIDs)) {
          auto position = expr.cast<AffineDimExpr>().getPosition();
          if (iterVars[position] != "") {
            continue;
          }
          assert(iteratorTypes[position] == "reduction");
          numReductionLoops++;
          iterVars[position] = getIteratorName(position);
          emitForLoop(iterVars[position], size, sizeID);
        }
      }
    }
    // 3. emit load
    auto &block = *(op.getBlock());
    for (auto [idx, operand] : llvm::enumerate(operands)) {
      std::string name = "in" + std::to_string(idx);
      out.indent(indent) << "double " << name << " = ";
      out << valueNameMap[operand];
      emitAffineMap(indexingMaps[idx], iterVars);
      out << ";\n";
      auto blockArg = block.getArgument(idx);
      valueNameMap[blockArg] = name;
    }
    // 4. emit body
    for (auto [idx, op] : llvm::enumerate(block)) {
      if (auto addFOp = dyn_cast_or_null<arith::AddFOp>(&op)) {
        emitArithAddFOp(addFOp, idx);
      } else if (auto mulFOp = dyn_cast_or_null<arith::MulFOp>(&op)) {
        emitArithMulFOp(mulFOp, idx);
      } else if (auto yieldOp = dyn_cast_or_null<linalg::YieldOp>(&op)) {
        for (auto [idx, output] : llvm::enumerate(outputs)) {
          auto affineMap = indexingMaps[inputs.size() + idx];
          out.indent(indent) << valueNameMap[output];
          emitAffineMap(affineMap, iterVars);
          out << " = " << valueNameMap[yieldOp.getOperand(idx)] << ";\n";
        }
      } else {
        out.indent(indent) << "// Failed: Emit C for op: " << op.getName()
                           << "\n";
      }
    }
    emitForLoopsEnd(numReductionLoops);
    emitForLoopsEnd(numParallelLoops);
    for (auto [result, output] : llvm::zip(results, outputs)) {
      // Today, I was really really shocked,
      // one can not write program like this.
      // In this statement, the reference of 
      // map[out] is getted, and then the map[res]
      // is added. The addition would make
      // the map expanded and then the map[out]
      // is moved, therefore it makes origin
      // reference invalid.
      // So, the map[res] is assigned to invalid
      // value!
      // valueNameMap[result] = valueNameMap[output];
      // C++ loose, Rust win!
      auto name = valueNameMap[output];
      valueNameMap[result] = name;
    }
  }
  void emitForLoop(std::string iterName, int64_t size, int64_t sizeID) {
    out.indent(indent) << "for (";
    // initial
    out << "int " << iterName << " = 0; ";
    // condition
    out << iterName << " < ";
    emitConstantOrVaribaleSize(size, sizeID);
    out << "; ";
    // increment
    out << iterName << "++) {\n";
    indent += 2;
  }
  void emitForLoopsEnd(int numLoops) {
    while (numLoops--) {
      indent -= 2;
      out.indent(indent) << "}\n";
    }
  }
  // linalg affine map for load or store
  void emitAffineMap(AffineMap map, PositionIterVariableMapTy &posIterMap) {
    for (auto expr : map.getResults()) {
      auto position = expr.dyn_cast<AffineDimExpr>().getPosition();
      out << "[";
      out << posIterMap[position];
      out << "]";
    }
  }
  void emitAffineMap(AffineMap map, SmallVector<std::string> &iterVars) {
    for (auto expr : map.getResults()) {
      auto position = expr.dyn_cast<AffineDimExpr>().getPosition();
      out << "[";
      out << iterVars[position];
      out << "]";
    }
  }
  void emitIterVars(SmallVector<std::string> &iterVars) {
    for (auto iterVar : iterVars) {
      out << "[";
      out << iterVar;
      out << "]";
    }
  }
  // emit arith add and mul
  void emitArithAddFOp(arith::AddFOp op, int64_t varIdx) {
    std::string name = "var" + std::to_string(varIdx);
    out.indent(indent) << "double " << name << " = ";
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();
    out << valueNameMap[lhs] << " + ";
    out << valueNameMap[rhs] << ";\n";
    valueNameMap[op.getResult()] = name;
  }
  void emitArithMulFOp(arith::MulFOp op, int64_t varIdx) {
    std::string name = "var" + std::to_string(varIdx);
    out.indent(indent) << "double " << name << " = ";
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();
    out << valueNameMap[lhs] << " * ";
    out << valueNameMap[rhs] << ";\n";
    valueNameMap[op.getResult()] = name;
  }

  inline std::string getIteratorName(unsigned position) {
    return "c" + std::to_string(position);
  }
};
} // namespace

namespace mlir::tc::codegen {
// void emitLinalgToC(ModuleOp op, llvm::raw_ostream &out) {
//   SymbolTable symTable(op);
//   auto funcOps = op.getOps<func::FuncOp>();
//   for (auto funcOp : funcOps) {
//     EmitCImpl(funcOp, out).emit();
//   }
// }
} // namespace mlir::tc::codegen
