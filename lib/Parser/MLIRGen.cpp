#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Verifier.h"

#include "TCC/Dialect/ContractionOps.h"
#include "TCC/MLIRGen.h"

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
  llvm::DenseMap<StringRef, Value> TreeTensorValueMap;
  llvm::DenseMap<StringRef, Value> InputTensorValueMap;
  llvm::DenseMap<Value, IndexesTy> ValueIndexesMap;

public:
  ModuleOp MLIRGen() {
    // create mlir module
    M = ModuleOp::create(Loc);
    OB.setInsertionPointToEnd(M.getBody());
    // create mlir function
    InputTensorsTy &ITs = CF.getInputTensors();
    ContractionTreesTy &OTs = CF.getOutputTensors();
    SmallVector<Type> InputTypes;
    for (auto &IT : ITs) {
      SmallVector<int64_t> Shape(IT->getDimension(), -1);
      SmallVector<int64_t> Sizes;
      for (auto Sz : IT->getSizes())
        Sizes.emplace_back(CF.getInputSizeOf(Sz));
      Attribute EncodingAttr = OB.getI64ArrayAttr(Sizes);
      Type T = RankedTensorType::get(Shape, OB.getF64Type(), EncodingAttr);
      InputTypes.emplace_back(T);
    }
    SmallVector<Type> ResultTypes;
    SmallVector<int64_t> WriteBacks;
    for (auto &CT : OTs) {
      SmallVector<int64_t> Shape(CT->getDimension(), -1);
      SmallVector<int64_t> Sizes;
      for (auto Sz : CT->getSizes())
        Sizes.emplace_back(CF.getInputSizeOf(Sz));
      Attribute EncodingAttr = OB.getI64ArrayAttr(Sizes);
      Type T = RankedTensorType::get(Shape, OB.getF64Type(), EncodingAttr);
      ResultTypes.emplace_back(T);
      int64_t ITIdx = ITs.size() - 1;
      for (; ITIdx >= 0; ITIdx--) {
        if (CT->getName() == ITs[ITIdx]->getName())
          break;
      }
      WriteBacks.emplace_back(ITIdx);
    }
    auto FT = OB.getFunctionType(InputTypes, ResultTypes);
    auto FuncAttr = SmallVector<NamedAttribute>(
        {OB.getNamedAttr("num_input_sizes",
                         OB.getI64IntegerAttr(CF.getInputSizes().size())),
         OB.getNamedAttr("wirtebacks", OB.getI64ArrayAttr(WriteBacks))});
    auto F = OB.create<func::FuncOp>(Loc, CF.getName(), FT, FuncAttr);
    auto EntryB = F.addEntryBlock();
    OB.setInsertionPointToEnd(EntryB);
    // Map input tensor to mlir value
    for (size_t i = 0; i < ITs.size(); i++) {
      InputTensorValueMap[ITs[i]->getName()] = EntryB->getArgument(i);
    }
    // Generate Contraction Dialect of trees
    auto &CTs = CF.getContractionTrees();
    for (auto &CT : CTs) {
      generateContractionTree(*CT.get());
    }
    // create mlir return
    // writebacks point to the argument tensor
    SmallVector<Value> ReturnValues;
    for (auto &CT : OTs) {
      ReturnValues.emplace_back(TreeTensorValueMap[CT->getName()]);
    }
    OB.create<func::ReturnOp>(Loc, TypeRange(), ReturnValues);
    return M;
  }

private:
  void generateContractionTree(ContractionTree &CT) {
    ValueIndexesMap.clear();
    auto Value = generateNode(*(CT.getRoot()));
    TreeTensorValueMap[CT.getName()] = Value;
  }

  Value generateNode(Node &N) {
    if (N.Type == Node::Root) {
      return generateRootNode(N);
    } else if (N.Type == Node::Addition) {
      return generateArithNode<AddOp>(N);
    } else if (N.Type == Node::Multiplication) {
      return generateArithNode<MulOp>(N);
    } else if (N.Type == Node::InputAsLeaf || N.Type == Node::TreeAsLeaf) {
      return generateIndexedTensorNode(N);
    } else {
      assert(N.Type == Node::Contraction);
      return generateContractionNode(N);
    }
  }
  static SmallVector<int64_t> getTensorSizes(Value V) {
    assert(V.getType().isa<RankedTensorType>());
    RankedTensorType T = V.getType().cast<RankedTensorType>();
    auto Attr = T.getEncoding().cast<ArrayAttr>();
    SmallVector<int64_t> Sizes;
    for (auto &I64 : Attr) {
      Sizes.emplace_back(I64.cast<IntegerAttr>().getValue().getSExtValue());
    }
    return Sizes;
  }
  Value generateRootNode(Node &N) {
    Value V = generateNode(N.getSingleChild());
    IndexesTy &InIndexes = ValueIndexesMap[V];
    IndexesTy &OutIndexes = N.getIndexes();
    assert(InIndexes.size() == OutIndexes.size());
    SmallVector<int64_t> AxesMap;
    for (auto Idx : InIndexes) {
      for (size_t Axis = 0; Axis < OutIndexes.size(); Axis++) {
        if (Idx == OutIndexes[Axis]) {
          AxesMap.emplace_back(Axis);
          break;
        }
      }
    }
    assert(InIndexes.size() == AxesMap.size());
    auto InSizes = getTensorSizes(V);
    SmallVector<int64_t> OutSizes(InSizes.size());
    for (size_t InAxis = 0; InAxis < AxesMap.size(); InAxis++) {
      auto OutAxis = AxesMap[InAxis];
      OutSizes[OutAxis] = InSizes[InAxis];
    }
    SmallVector<int64_t> OutShape(OutSizes.size(), -1);
    auto OutAttr = OB.getI64ArrayAttr(OutSizes);
    auto OutType = RankedTensorType::get(OutShape, OB.getF64Type(), OutAttr);
    auto AxesMapAttr = OB.getI64ArrayAttr(AxesMap);
    V = OB.create<PermuteOp>(Loc, OutType, V, AxesMapAttr);
    return V;
  }
  Value generateContractionNode(Node &N) {
    Value V = generateNode(N.getSingleChild());
    auto InSizes = getTensorSizes(V);
    IndexesTy &InIndexes = ValueIndexesMap[V];
    IndexesTy &CtIndexes = N.getIndexes();
    IndexesTy OutIndexes;
    SmallVector<int64_t> OutSizes;
    SmallVector<int64_t> ContractedAxes;
    for (size_t Axis = 0; Axis < InIndexes.size(); Axis++) {
      auto &Idx1 = InIndexes[Axis];
      if (llvm::none_of(CtIndexes, [&](auto &Idx2) { return Idx1 == Idx2; })) {
        OutSizes.emplace_back(InSizes[Axis]);
        OutIndexes.emplace_back(Idx1);
      } else {
        ContractedAxes.emplace_back(Axis);
      }
    }
    SmallVector<int64_t> OutShape(OutSizes.size(), -1);
    auto OutAttr = OB.getI64ArrayAttr(OutSizes);
    auto OutType = RankedTensorType::get(OutShape, OB.getF64Type(), OutAttr);
    auto AxesAtrr = OB.getI64ArrayAttr(ContractedAxes);
    V = OB.create<ContractOp>(Loc, OutType, V, AxesAtrr);
    ValueIndexesMap[V] = OutIndexes;
    return V;
  }
  template <class T> Value generateArithNode(Node &N) {
    auto &Childs = N.getChilds();
    SmallVector<Value> Operands;
    SmallVector<int64_t> OutSizes;
    IndexesTy OutIndexes;
    SmallVector<SmallVector<int64_t>> OutAxesMaps(Childs.size());
    for (size_t Ci = 0; Ci < Childs.size(); Ci++) {
      auto &C = Childs[Ci];
      Value V = generateNode(*C);
      Operands.emplace_back(V);
      auto Sizes = getTensorSizes(V);
      auto &Indexes = ValueIndexesMap[V];
      auto &AxesMap = OutAxesMaps[Ci];
      for (size_t Axis = 0; Axis < Indexes.size(); Axis++) {
        size_t OutAxis = 0;
        for (; OutAxis < OutIndexes.size(); OutAxis++) {
          if (Indexes[Axis] == OutIndexes[OutAxis])
            break;
        }
        if (OutAxis == OutIndexes.size()) {
          OutSizes.emplace_back(Sizes[Axis]);
          OutIndexes.emplace_back(Indexes[Axis]);
        }
        AxesMap.emplace_back(OutAxis);
      }
    }
    SmallVector<int64_t> OutShape(OutSizes.size(), -1);
    auto OutAttr = OB.getI64ArrayAttr(OutSizes);
    auto OutType = RankedTensorType::get(OutShape, OB.getF64Type(), OutAttr);
    SmallVector<Attribute, 4> AxesMapAttrs;
    for (auto &AxesMap : OutAxesMaps) {
      AxesMapAttrs.emplace_back(OB.getI64ArrayAttr(AxesMap));
    }
    auto AxesMapsAttr = OB.getArrayAttr(AxesMapAttrs);
    Value V = OB.create<T>(Loc, OutType, Operands, AxesMapsAttr);
    ValueIndexesMap[V] = OutIndexes;
    return V;
  }
  Value generateIndexedTensorNode(Node &N) {
    Value V;
    if (N.Type == Node::InputAsLeaf) {
      auto IT = N.getInputTensor();
      V = InputTensorValueMap[IT->getName()];
    } else {
      assert(N.Type == Node::TreeAsLeaf);
      auto CT = N.getContractionTensor();
      V = TreeTensorValueMap[CT->getName()];
    }
    ValueIndexesMap[V] = N.getIndexes();
    return V;
  }
};

OwningOpRef<ModuleOp> MLIRGen(MLIRContext &Ctx, ContractionForest &CF) {
  return MLIRGenImpl(Ctx, CF).MLIRGen();
}
} // namespace contraction
} // namespace mlir
