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

  void emit() { emitImpl(); }

private:
  int64_t indent = 0;

  using ValueNameMapTy = DenseMap<Value, std::string>;
  ValueNameMapTy valueNameMap;
  using SizeIDSymbolsTy = SmallVector<std::string>;
  SizeIDSymbolsTy sizeIDSymbols;
  ArrayRef<int64_t> lbs;
  ArrayRef<int64_t> ubs;

  void emitImpl() {
    auto sizeRanges = getSizesRanges(funcOp);
    lbs = sizeRanges.first;
    ubs = sizeRanges.second;
    sizeIDSymbols.reserve(ubs.size());
    for (auto ub : ubs)
      sizeIDSymbols.emplace_back(std::to_string(ub));
    emitFunctionType();
    emitFunctionBody();
  }

  void emitFunctionType() {
    auto declare = ("void " + funcOp.getName() + "(").str();
    out << declare;
    indent += declare.size();
    emitArgumentTypes();
    emitResultTypes();
    out << ")";
  }

  // emit local array
  int64_t localIdx = 0;
  void emitLocalArrayForTensor(Value value) {
    if (valueNameMap.count(value)) {
      return;
    }
    std::string name = "local_" + std::to_string(localIdx++);
    out.indent(indent) << "double " << name;
    auto sizeIDs = getTensorSizeIDs(value);
    for (auto sizeID : sizeIDs)
      out << "[" << sizeIDSymbols[sizeID] << "]";
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

  void emitArgumentTypes() {
    auto &block = funcOp.front();
    auto inputs = block.getArguments();
    for (auto [idx, input] : llvm::enumerate(inputs)) {
      auto name = getArgName(idx);
      valueNameMap[input] = name;
      if (input.getType().isIndex()) {
        sizeIDSymbols[idx] = name;
        out << "int " << name;
      } else
        emitTensorArgument(input, name, true);
      if (idx + 1 != inputs.size()) {
        out << ",\n";
        out.indent(indent);
      }
    }
  }

  void emitResultTypes() {
    auto results = returnOp.getOperands();
    for (auto [idx, result] : llvm::enumerate(results)) {
      out << ",\n";
      out.indent(indent);
      auto name = getResultName(idx);
      valueNameMap[result] = name;
      emitTensorArgument(result, name, true);
    }
  }

  void emitTensorArgument(Value value, std::string name,
                          bool isRestrict = false) {
    out << "double " << name;
    auto sizeIDs = getTensorSizeIDs(value);
    for (auto [idx, sizeID] : llvm::enumerate(sizeIDs)) {
      out << "[";
      if (idx == 0 && isRestrict)
        out << "const restrict ";
      out << sizeIDSymbols[sizeID] << "]";
    }
  }

  void emitFunctionBody() {
    out << "\n{\n";
    out << "#pragma scop\n";
    auto numInputSizes = getNumInputSizes(funcOp);
    if (numInputSizes) {
      out << "if (";
      llvm::interleave(
          llvm::seq<int64_t>(0, numInputSizes), out,
          [&](int64_t sizeID) {
            out << lbs[sizeID] << " <= " << sizeIDSymbols[sizeID];
            out << " && ";
            out << sizeIDSymbols[sizeID] << " <= " << ubs[sizeID];
          },
          " && ");
      out << ")";
    }
    out << "{\n";
    indent = 2;
    for (auto &op : block) {
      LLVM_DEBUG(llvm::dbgs() << "Emitting " << op.getName() << "\n";
                 llvm::dbgs() << "\t" << op << "\n";);
      if (auto cstOp = dyn_cast_or_null<arith::ConstantFloatOp>(&op)) {
        emitArithConstantOp(cstOp);
      } else if (auto linalgOp = dyn_cast_or_null<linalg::GenericOp>(&op)) {
        emitLinalgGenericOp(linalgOp);
      } else if (auto fillOp = dyn_cast_or_null<linalg::FillOp>(&op)) {
      } else if (auto emptyOp = dyn_cast_or_null<tensor::EmptyOp>(&op)) {
      } else if (auto returnOp = dyn_cast_or_null<ReturnOp>(&op)) {
      } else {
        out.indent(indent) << "// Failed: Emit C for op: ";
        out << op.getName() << "\n";
      }
    }
    out << "}\n";
    out << "#pragma endscop\n";
    out << "}\n";
  }

  // emit arith constant
  void emitArithConstantOp(arith::ConstantFloatOp op) {
    SmallVector<char> value;
    op.value().toString(value);
    valueNameMap[op.getResult()] = std::string(value.begin(), value.end());
  }

  // emit tensor empty op
  void emitTensorEmptyOp(tensor::EmptyOp op) {
    out.indent(indent) << "// From Tensor EmptyOp\n";
    emitLocalArrayForTensor(op.getResult());
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
    LLVM_DEBUG(out.indent(indent) << "// From Linalg GenericOp\n";);
    auto inputs = op.getInputs();
    auto outputs = op.getOutputs();
    auto results = op.getResults();
    auto indexingMaps = op.getIndexingMapsArray();
    auto iteratorTypes = op.getIteratorTypesArray();
    // 1. Create local array for results
    for (auto result : results)
      emitLocalArrayForTensor(result);
    // 2. emit parallel loop
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
    // 3. Copy output to result
    for (auto [idx, result] : llvm::enumerate(results)) {
      auto output = outputs[idx];
      if (!llvm::isa<tensor::EmptyOp>(output.getDefiningOp())) {
        out.indent(indent) << valueNameMap[result];
        auto affineMap = indexingMaps[inputs.size() + idx];
        auto arrayUsage = getArrayUsage(affineMap, iterVars);
        out << arrayUsage;
        out << " = ";
        if (auto fillOp = output.getDefiningOp<linalg::FillOp>()) {
          out << valueNameMap[fillOp.getInputs()[0]];
        } else {
          out << valueNameMap[outputs[idx]];
          out << arrayUsage;
        }
        out << ";\n";
      }
    }
    // 4. Emit reduction loop
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
    // 5. Map block arguments to array usages
    auto &block = *(op.getBlock());
    for (auto [idx, input] : llvm::enumerate(inputs)) {
      auto arrayUsageExpr = valueNameMap[input];
      arrayUsageExpr += getArrayUsage(indexingMaps[idx], iterVars);
      auto blockArg = block.getArgument(idx);
      valueNameMap[blockArg] = arrayUsageExpr;
    }
    // 6. Map yieldOp values to array usages
    auto yieldOp = cast<linalg::YieldOp>(block.getTerminator());
    for (auto [idx, result] : llvm::enumerate(results)) {
      auto arrayUsageExpr = valueNameMap[result];
      auto affineMap = indexingMaps[inputs.size() + idx];
      arrayUsageExpr += getArrayUsage(affineMap, iterVars);
      auto blockArg = block.getArgument(inputs.size() + idx);
      valueNameMap[blockArg] = arrayUsageExpr;
      valueNameMap[yieldOp.getOperand(idx)] = arrayUsageExpr;
    }
    // 7. emit body
    for (auto [idx, op] : llvm::enumerate(block)) {
      if (auto addFOp = dyn_cast_or_null<arith::AddFOp>(&op)) {
        emitArithAddFOp(addFOp, idx);
      } else if (auto mulFOp = dyn_cast_or_null<arith::MulFOp>(&op)) {
        emitArithMulFOp(mulFOp, idx);
      } else if (auto yieldOp = dyn_cast_or_null<linalg::YieldOp>(&op)) {
      } else {
        out.indent(indent) << "// Failed: Emit C for op: " << op.getName()
                           << "\n";
      }
    }
    emitForLoopsEnd(numReductionLoops);
    emitForLoopsEnd(numParallelLoops);
  }
  void emitForLoop(std::string iterName, int64_t size, int64_t sizeID) {
    out.indent(indent) << "for (";
    // initial
    out << "int " << iterName << " = 0; ";
    // condition
    out << iterName << " < ";
    out << sizeIDSymbols[sizeID];
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
  std::string getArrayUsage(AffineMap map, SmallVector<std::string> &iterVars) {
    std::string usage;
    for (auto expr : map.getResults()) {
      auto position = expr.dyn_cast<AffineDimExpr>().getPosition();
      usage += "[" + iterVars[position] + "]";
    }
    return usage;
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
    auto result = op.getResult();
    if (valueNameMap.count(result)) {
      out.indent(indent) << valueNameMap[result];
    } else {
      std::string name = "var" + std::to_string(varIdx);
      valueNameMap[result] = name;
      out.indent(indent) << "double " << name;
    }
    out << " = ";
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();
    out << valueNameMap[lhs] << " + ";
    out << valueNameMap[rhs] << ";\n";
  }
  void emitArithMulFOp(arith::MulFOp op, int64_t varIdx) {
    auto result = op.getResult();
    if (valueNameMap.count(result)) {
      out.indent(indent) << valueNameMap[result];
    } else {
      std::string name = "var" + std::to_string(varIdx);
      valueNameMap[result] = name;
      out.indent(indent) << "double " << name;
    }
    out << " = ";
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();
    out << valueNameMap[lhs] << " * ";
    out << valueNameMap[rhs] << ";\n";
  }

  inline std::string getIteratorName(unsigned position) {
    return "c" + std::to_string(position);
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
};
} // namespace

namespace mlir::tc::codegen {
void emitLinalgToC(ModuleOp op, llvm::raw_ostream &out) {
  SymbolTable symTable(op);
  auto funcOps = op.getOps<func::FuncOp>();
  for (auto funcOp : funcOps) {
    EmitCImpl(funcOp, out).emit();
  }
}
} // namespace mlir::tc::codegen
