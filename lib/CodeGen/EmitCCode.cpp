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
    emitArgumentTypes();
    emitResultTypes();
    out << ")\n";
  }

  void emitArgumentTypes() {
    auto inputs = block.getArguments();
    for (auto [idx, input] : llvm::enumerate(inputs)) {
      if (idx != 0) {
        out << ",\n";
        out.indent(indent);
      }
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

  void emitResultTypes() {
    // TODO
  }

  void emitFunctionBody() {
    out << "{\n";
    indent = 2;
    for (auto &op : block) {
      if (auto cstOp = dyn_cast_or_null<arith::ConstantFloatOp>(&op)) {
        emitArithConstantOp(cstOp);
      } else if (auto linalgOp = dyn_cast_or_null<linalg::GenericOp>(&op)) {
        emitLinalgGenericOp(linalgOp);
      } else if (auto emptyOp = dyn_cast_or_null<tensor::EmptyOp>(&op)) {
        // Ignore
      } else {
        out.indent(indent);
        out << "// Failed: Emit C for op: " << op.getName() << "\n";
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
        out.indent(indent) << "// Failed: Emit C for op: " << op.getName() << "\n";
      }
    }
    for (auto &__ : posIterMap) {
      (void)__;
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
  // emit arith add and mul
  void emitArithAddFOp(arith::AddFOp op, int64_t varIdx) {
    std::string name = "var" + std::to_string(varIdx);
    out.indent(indent) << "double " << name << " = ";
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();
    llvm::errs() << valueNameMap[lhs] << " + ";
    llvm::errs() << valueNameMap[rhs] << ";\n";
    valueNameMap[op.getResult()] = name;
  }
  void emitArithMulFOp(arith::MulFOp op, int64_t varIdx) {
    std::string name = "var" + std::to_string(varIdx);
    out.indent(indent) << "double " << name << " = ";
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();
    llvm::errs() << valueNameMap[lhs] << " * ";
    llvm::errs() << valueNameMap[rhs] << ";\n";
    valueNameMap[op.getResult()] = name;
  }
};
} // namespace

namespace mlir::tc::codegen {
using ValueNameMapTy = DenseMap<Value, std::string>;

void emitMalloc(Value value, ValueNameMapTy &valueNameMap,
                int64_t &mallocName) {
  std::string name = "alloc" + std::to_string(mallocName++);
  llvm::errs() << "double *" << name << " = "
               << "malloc(sizeof(double)";
  auto [shape, sizeIDs] = getTensorSizeInfos(value);
  for (auto [size, sizeID] : llvm::zip(shape, sizeIDs)) {
    llvm::errs() << " * ";
    if (size >= 0) {
      llvm::errs() << size;
    } else {
      llvm::errs() << "arg" << sizeID;
    }
  }
  llvm::errs() << ");\n";
  valueNameMap[value] = name;
}

void emitLocalArray(Value value, ValueNameMapTy &valueNameMap,
                    int64_t &localIdx) {
  std::string name = "local_" + std::to_string(localIdx++);
  llvm::errs() << "double ";
  llvm::errs() << name;
  auto [shape, sizeIDs] = getTensorSizeInfos(value);
  for (auto [size, sizeID] : llvm::zip(shape, sizeIDs)) {
    llvm::errs() << "[";
    if (size >= 0) {
      llvm::errs() << size;
    } else {
      llvm::errs() << "arg" << sizeID;
    }
    llvm::errs() << "]";
  }
  llvm::errs() << ";\n";
  valueNameMap[value] = name;
}

void emitTensorArgument(Value value, ValueNameMapTy &valueNameMap,
                        int64_t idx) {
  std::string name = "arg" + std::to_string(idx);
  llvm::errs() << "double ";
  llvm::errs() << name;
  auto [shape, sizeIDs] = getTensorSizeInfos(value);
  for (auto [size, sizeID] : llvm::zip(shape, sizeIDs)) {
    llvm::errs() << "[";
    if (size >= 0) {
      llvm::errs() << size;
    } else {
      llvm::errs() << "arg" << sizeID;
    }
    llvm::errs() << "]";
  }
  valueNameMap[value] = name;
}

void emitAffineMap(AffineMap map,
                   DenseMap<int64_t, std::string> &posIterNameMap) {
  for (auto expr : map.getResults()) {
    auto pos = expr.dyn_cast<AffineDimExpr>().getPosition();
    llvm::errs() << "[";
    llvm::errs() << posIterNameMap[pos];
    llvm::errs() << "]";
  }
}

void emitCCode(ModuleOp moduleOp) {
  DenseMap<Value, std::string> valueNameMap;
  SymbolTable symTable(moduleOp);
  func::FuncOp funcOp = symTable.lookup<func::FuncOp>("CT");
  assert(funcOp);
  EmitCImpl(funcOp, llvm::errs()).emit();
  llvm::errs() << "Emit C Code\n";
  auto argumentTypes = funcOp.getArgumentTypes();
  llvm::errs() << "void CT(";
  for (auto [idx, type] : llvm::enumerate(argumentTypes)) {
    if (idx != 0) {
      llvm::errs() << ", ";
    }
    if (type.isIndex()) {
      std::string name = "arg" + std::to_string(idx);
      llvm::errs() << "size_t ";
      llvm::errs() << name;
      valueNameMap[funcOp.getArgument(idx)] = name;
    } else {
      emitTensorArgument(funcOp.getArgument(idx), valueNameMap, idx);
    }
  }
  llvm::errs() << ") {\n";
  int64_t allocaName = 0;
  int64_t cstIdx = 0;
  for (auto &op : funcOp.front()) {
    if (auto emptyOp = dyn_cast_or_null<tensor::EmptyOp>(&op)) {
      // Ignore
    } else if (auto cstOp = dyn_cast_or_null<arith::ConstantFloatOp>(&op)) {
      std::string name = "cst" + std::to_string(cstIdx++);
      llvm::errs() << "double " << name << " = ";
      SmallVector<char> buffer;
      cstOp.value().toString(buffer);
      llvm::errs() << buffer;
      llvm::errs() << ";\n";
      valueNameMap[cstOp.getResult()] = name;
    } else if (auto linalgOp = dyn_cast_or_null<linalg::GenericOp>(&op)) {
      llvm::errs() << "// TODO: Emit C for op: " << op.getName() << "\n";
      DenseMap<int64_t, std::string> posIterNameMap;
      auto iterTypes = linalgOp.getIteratorTypesArray();
      auto indexingMaps = linalgOp.getIndexingMapsArray();
      auto resultMap = indexingMaps[indexingMaps.size() - 1];
      auto result = linalgOp.getResult(0);
      emitLocalArray(result, valueNameMap, allocaName);
      auto [shape, sizeIDs] = getTensorSizeInfos(result);
      for (auto [idx, sizeID] : llvm::enumerate(sizeIDs)) {
        std::string name = "i" + std::to_string(idx);
        llvm::errs() << "for (";
        llvm::errs() << "int64_t " << name << " = 0; ";
        llvm::errs() << name << " < ";
        if (shape[idx] > 0) {
          llvm::errs() << shape[idx];
        } else {
          llvm::errs() << "arg" << sizeID;
        }
        llvm::errs() << "; ";
        llvm::errs() << name << "++)";
        llvm::errs() << "\n";
        auto expr = resultMap.getResult(idx).dyn_cast<AffineDimExpr>();
        auto pos = expr.getPosition();
        posIterNameMap[pos] = name;
      }
      llvm::errs() << "{\n";
      auto inputs = linalgOp.getInputs();
      auto block = linalgOp.getBlock();
      for (auto [idx, input] : llvm::enumerate(inputs)) {
        std::string name = "in" + std::to_string(idx);
        llvm::errs() << "double " << name << " = ";
        llvm::errs() << valueNameMap[input];
        emitAffineMap(indexingMaps[idx], posIterNameMap);
        llvm::errs() << ";\n";
        auto blockArg = block->getArgument(idx);
        valueNameMap[blockArg] = name;
      }
      for (auto [idx, op] : llvm::enumerate(*block)) {
        if (auto addfOp = dyn_cast_or_null<arith::AddFOp>(&op)) {
          std::string name = "var" + std::to_string(idx);
          llvm::errs() << "double " << name << " = ";
          auto lhs = addfOp.getLhs();
          auto rhs = addfOp.getRhs();
          llvm::errs() << valueNameMap[lhs] << " + ";
          llvm::errs() << valueNameMap[rhs] << ";\n";
          valueNameMap[addfOp.getResult()] = name;
        } else if (auto mulfOp = dyn_cast_or_null<arith::MulFOp>(&op)) {
          std::string name = "var" + std::to_string(idx);
          llvm::errs() << "double " << name << " = ";
          auto lhs = mulfOp.getLhs();
          auto rhs = mulfOp.getRhs();
          llvm::errs() << valueNameMap[lhs] << " * ";
          llvm::errs() << valueNameMap[rhs] << ";\n";
          valueNameMap[mulfOp.getResult()] = name;
        } else if (auto yieldOp = dyn_cast_or_null<linalg::YieldOp>(&op)) {
          auto input = yieldOp.getOperand(0);
          llvm::errs() << valueNameMap[result];
          emitAffineMap(resultMap, posIterNameMap);
          llvm::errs() << " = ";
          llvm::errs() << valueNameMap[input];
          llvm::errs() << ";\n";
        } else {
          llvm::errs() << "// TODO: Emit C for op: " << op.getName() << "\n";
        }
      }
      llvm::errs() << "}\n";
    } else if (auto blasOp = dyn_cast_or_null<tblas::GenericOp>(&op)) {
      auto result = blasOp.getResult();
      emitLocalArray(result, valueNameMap, allocaName);
      llvm::errs() << "blas_generic();\n";
    } else {
      llvm::errs() << "// TODO: Emit C for op: " << op.getName() << "\n";
    }
  }
  llvm::errs() << "}\n";
}
} // namespace mlir::tc::codegen
