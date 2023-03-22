#include "TCC/CodeGen.h"

#include "TCC/Dialect/TBLAS/IR/TBLASOps.h"

#include "TCC/Dialect/TC/IR/TCOps.h"

#include "TCC/Dialect/TC/Utils/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;
using namespace mlir::tc;

namespace {
class EmitCImpl {
  func::FuncOp funcOp;
  Block &block;
  func::ReturnOp returnOp;
  llvm::raw_ostream &out;

public:
  EmitCImpl(func::FuncOp funcOp, llvm::raw_ostream &out)
      : funcOp(funcOp), block(funcOp.front()), out(out) {
    returnOp = dyn_cast<func::ReturnOp>(block.getTerminator());
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
  using SizeIDNameMapTy = DenseMap<int64_t, std::string>;
  SizeIDNameMapTy sizeIDNameMap;
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
      std::string name;
      auto nameAttr = funcOp.getArgAttr(idx, "func.name");
      if (nameAttr) {
        name = nameAttr.cast<StringAttr>().getValue();
      } else {
        name = "arg" + std::to_string(idx);
      }
      if (input.getType().isIndex()) {
        emitIndexArgument(input, idx, name);
      } else {
        emitTensorArgument(input, name);
      }
      valueNameMap[input] = name;
    }
  }

  void emitIndexArgument(Value value, int64_t sizeID, std::string name) {
    out << "size_t " << name;
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

  void emitResultTypes(bool &newLine) {
    // TODO: use writebacks
    // TODO: get result name
    for (auto [idx, input] : llvm::enumerate(returnOp.getOperands())) {
      if (newLine) {
        out << ",\n";
        out.indent(indent);
      }
      newLine = true;
      std::string name = "result" + std::to_string(idx);
      emitTensorArgument(input, name);
      valueNameMap[input] = name;
    }
  }

  void emitFunctionBody() {
    out << "{\n";
    indent = 2;
    for (auto &op : block) {
      if (auto cstOp = dyn_cast_or_null<arith::ConstantFloatOp>(&op)) {
        emitArithConstantOp(cstOp);
      } else if (auto linalgOp = dyn_cast_or_null<linalg::GenericOp>(&op)) {
        emitLinalgGenericOp(linalgOp);
      } else if (auto blasOp = dyn_cast_or_null<tblas::GenericOp>(&op)) {
        out.indent(indent) << "// Failed: Emit C for op: " << op.getName()
                           << "\n";
      } else if (auto emptyOp = dyn_cast_or_null<tensor::EmptyOp>(&op)) {
        // Ignore
      } else if (auto returnOp = dyn_cast_or_null<func::ReturnOp>(&op)) {
        // Ignore
      } else {
        out.indent(indent) << "// Failed: Emit C for op: " << op.getName()
                           << "\n";
      }
    }
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

  // emit linalg generic op
  using PositionIterVariableMapTy = DenseMap<int64_t, std::string>;
  void emitLinalgGenericOp(linalg::GenericOp op) {
    out.indent(indent) << "// Linalg GenericOp: " << op->getName() << "\n";
    // create alloc
    auto inputs = op.getInputs();
    auto results = op.getResults();
    for (auto result : op.getResults()) {
      emitLocalArrayForTensor(result);
    }
    PositionIterVariableMapTy posIterMap;
    auto iterTypes = op.getIteratorTypesArray();
    auto indexingMaps = op.getIndexingMapsArray();
    // emit for loop
    for (auto [idx, result] : llvm::enumerate(results)) {
      auto affineMap = indexingMaps[inputs.size() + idx];
      auto exprs = affineMap.getResults();
      auto [shape, sizeIDs] = getTensorSizeInfos(result);
      for (auto [size, sizeID, expr] : llvm::zip(shape, sizeIDs, exprs)) {
        auto position = expr.cast<AffineDimExpr>().getPosition();
        if (posIterMap.count(position)) {
          continue;
        }
        assert(iterTypes[position] == "parallel");
        out.indent(indent) << "for (";
        // initial
        auto iterName = "i" + std::to_string(position);
        out << "int64_t " << iterName << " = 0; ";
        // condition
        out << iterName << " < ";
        emitConstantOrVaribaleSize(size, sizeID);
        out << "; ";
        // tail
        out << iterName << "++) {\n";
        indent += 2;
        posIterMap[position] = iterName;
      }
    }
    // emit load
    auto &block = *(op.getBlock());
    for (auto [idx, input] : llvm::enumerate(inputs)) {
      std::string name = "in" + std::to_string(idx);
      out.indent(indent) << "double " << name << " = ";
      out << valueNameMap[input];
      emitAffineMap(indexingMaps[idx], posIterMap);
      out << ";\n";
      auto blockArg = block.getArgument(idx);
      valueNameMap[blockArg] = name;
    }
    // emit body
    for (auto [idx, op] : llvm::enumerate(block)) {
      if (auto addFOp = dyn_cast_or_null<arith::AddFOp>(&op)) {
        emitArithAddFOp(addFOp, idx);
      } else if (auto mulFOp = dyn_cast_or_null<arith::MulFOp>(&op)) {
        emitArithMulFOp(mulFOp, idx);
      } else if (auto yieldOp = dyn_cast_or_null<linalg::YieldOp>(&op)) {
        for (auto [idx, output] : llvm::enumerate(results)) {
          auto affineMap = indexingMaps[inputs.size() + idx];
          out.indent(indent) << valueNameMap[output];
          emitAffineMap(affineMap, posIterMap);
          out << " = " << valueNameMap[yieldOp.getOperand(idx)] << ";\n";
        }
      } else {
        out.indent(indent) << "// Failed: Emit C for op: " << op.getName()
                           << "\n";
      }
    }
    for (auto __ : llvm::seq<int>(0, posIterMap.size())) {
      (void)__;
      indent -= 2;
      indent = indent < 0 ? 0 : indent;
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
};
} // namespace

namespace mlir::tc::codegen {
void emitCCode(ModuleOp moduleOp) {
  SymbolTable symTable(moduleOp);
  auto funcOps = moduleOp.getOps<func::FuncOp>();
  for (auto funcOp : funcOps) {
    EmitCImpl(funcOp, llvm::errs()).emit();
  }
}
} // namespace mlir::tc::codegen
