#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Verifier.h"

#include "MLIRGen.h"
#include "TCDialect/ContractionOps.h"

using namespace contraction;

namespace mlir {
namespace contraction {
class MLIRGenImpl {
private:
  ContractionForest &CF;
  ModuleOp M;
  OpBuilder OB;
  Location Loc;

public:
  MLIRGenImpl(MLIRContext &Ctx, ContractionForest &CF)
      : CF(CF), OB(&Ctx), Loc(UnknownLoc::get(&Ctx)){};

private:
  llvm::DenseMap<int, Value> InputSizeValueMap;
  llvm::DenseMap<StringRef, Value> InputTensorValueMap;
  llvm::DenseMap<StringRef, Value> TreeValueMap;
  llvm::SmallVector<Value> IndexValueMap;

public:
  ModuleOp MLIRGen() {
    // create mlir module
    M = ModuleOp::create(Loc);
    OB.setInsertionPointToEnd(M.getBody());
    // create mlir function
    SizesTy &ISzs = CF.getInputSizes();
    InputTensorsTy &ITs = CF.getInputTensors();
    ContractionTreesTy &OTs = CF.getOutputTensors();
    SmallVector<Type> InputTypes(ISzs.size(), OB.getIndexType());
    for (auto &IT : ITs) {
      InputTypes.emplace_back(MemRefType::get(
          SmallVector<int64_t>(IT->getDimension(), -1), OB.getF64Type()));
    }
    SmallVector<Type> ResultTypes;
    for (auto &CT : OTs) {
      ResultTypes.emplace_back(RankedTensorType::get(
          SmallVector<int64_t>(CT->getDimension(), -1), OB.getF64Type()));
    }
    auto FuncType = OB.getFunctionType(InputTypes, ResultTypes);
    auto Func = OB.create<func::FuncOp>(Loc, CF.getName(), FuncType);
    auto EntryB = Func.addEntryBlock();
    OB.setInsertionPointToEnd(EntryB);
    // Map size and input tensor to mlir value
    for (size_t i = 0; i < ISzs.size(); i++) {
      InputSizeValueMap[ISzs[i]] = EntryB->getArgument(i);
    }
    for (size_t i = 0; i < ITs.size(); i++) {
      InputTensorValueMap[ITs[i]->getName()] =
          EntryB->getArgument(i + ISzs.size());
    }
    // Generate Contraction Dialect of trees
    auto &CTs = CF.getContractionTrees();
    for (auto &CT : CTs) {
      generateContractionTree(*CT.get());
    }
    // create mlir return
    SmallVector<Value> ValuesToReturn;
    for (auto &CT : OTs) {
      ValuesToReturn.emplace_back(TreeValueMap[CT->getUniqueName()]);
    }
    OB.create<func::ReturnOp>(Loc, ValuesToReturn);
    return M;
  }

private:
  void generateContractionTree(ContractionTree &CT) {
    IndexValueMap.clear();
    auto Value = generateNode(CT.getRoot());
    TreeValueMap[CT.getUniqueName()] = Value;
  }

  Value generateNode(std::unique_ptr<Node> &N) {
    if (N->Type == Node::Root) {
      return generateRootOrContractNode<ForOp>(N);
    } else if (N->Type == Node::Addition) {
      return generateArithNode<AddOp>(N);
    } else if (N->Type == Node::Multiplication) {
      return generateArithNode<MulOp>(N);
    } else if (N->Type == Node::InputAsLeaf || N->Type == Node::TreeAsLeaf) {
      return generateIndexedTensorNode(N);
    } else {
      assert(N->Type == Node::Contraction);
      return generateRootOrContractNode<ContractOp>(N);
    }
  }
  template <class T>
  Value generateRootOrContractNode(std::unique_ptr<Node> &N) {
    Block *OldB = OB.getInsertionBlock();
    SmallVector<Value, 4> RangesValue;
    SizesTy &Ranges = N->getSizes();
    for (auto Sz : Ranges) {
      Sz = CF.getUniqueInputSizeOf(Sz);
      RangesValue.emplace_back(InputSizeValueMap[Sz]);
    }
    auto ForOrContract = OB.create<T>(Loc, RangesValue);
    auto Body = ForOrContract.getBody();
    OB.setInsertionPointToEnd(Body);
    for (auto Idx : Body->getArguments()) {
      IndexValueMap.emplace_back(Idx);
    }
    auto ChildValue = generateNode(N->getChilds()[0]);
    OB.create<YieldOp>(Loc, ChildValue);
    OB.setInsertionPointToEnd(OldB);
    return ForOrContract;
  }
  template <class T> Value generateArithNode(std::unique_ptr<Node> &N) {
    ChildNodesTy &Childs = N->getChilds();
    assert(Childs.size() >= 2);
    auto LHS = generateNode(Childs[0]);
    for (size_t i = 1; i < Childs.size(); i++) {
      auto RHS = generateNode(Childs[i]);
      LHS = OB.create<T>(Loc, LHS, RHS);
    }
    return LHS;
  }
  Value generateIndexedTensorNode(std::unique_ptr<Node> &N) {
    Value TensorValue;
    if (N->Type == Node::InputAsLeaf) {
      auto IT = N->getInputTensor();
      TensorValue = InputTensorValueMap[IT->getName()];
    } else {
      assert(N->Type == Node::TreeAsLeaf);
      auto CT = N->getContractionTensor();
      TensorValue = TreeValueMap[CT->getUniqueName()];
    }
    SmallVector<Value> IndexValues;
    auto Indexes = N->getIndexes();
    for (auto I : Indexes) {
      IndexValues.emplace_back(IndexValueMap[I]);
    }
    return OB.create<IndexOp>(Loc, TensorValue, IndexValues);
  }
};

OwningOpRef<ModuleOp> MLIRGen(MLIRContext &Ctx, ContractionForest &CF) {
  return MLIRGenImpl(Ctx, CF).MLIRGen();
}
} // namespace contraction
} // namespace mlir
