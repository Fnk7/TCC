#include "TCC/CodeGen.h"

#include "TCC/Dialect/TC/IR/TCOps.h"

#include "TCC/Dialect/TC/Utils/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir::tc;

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

void emitAffineMap(AffineMap map, DenseMap<int64_t, std::string> &posIterNameMap) {
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
    } else {
      llvm::errs() << "// TODO: Emit C for op: " << op.getName() << "\n";
    }
  }
  llvm::errs() << "}\n";
}
} // namespace mlir::tc::codegen
