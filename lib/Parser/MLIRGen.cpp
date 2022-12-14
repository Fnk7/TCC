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

using llvm::cast;
using llvm::isa;

using llvm::errs;

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
  llvm::DenseMap<StringRef, Value> SymbolValueMap;

  using IndexedValueTy = std::pair<Value, IndexUsesTy>;

  auto getTensorTypeOfShape(ShapeTy &Shape) {
    SmallVector<int64_t> TShape;
    for (auto Size : Shape) {
      if (Size->isConstant()) {
        TShape.emplace_back(Size->getUpperBound());
      } else {
        TShape.emplace_back(-1);
      }
    }
    SmallVector<int64_t> Encoding;
    for (auto Size : Shape) {
      Encoding.emplace_back(Size->getUID());
    }
    auto EncodingAttr = OB.getI64ArrayAttr(Encoding);
    return RankedTensorType::get(TShape, OB.getF64Type(), EncodingAttr);
  }

public:
  ModuleOp MLIRGen() {
    // create mlir module
    M = ModuleOp::create(Loc);
    OB.setInsertionPointToEnd(M.getBody());
    // create mlir function
    InTensorsTy &ITs = CF.getInputTensors();
    CTreePtrsTy &OTs = CF.getOutputTensors();
    SmallVector<Type> InputTypes;
    for (auto &IT : ITs) {
      auto Ty = getTensorTypeOfShape(IT->getShape());
      InputTypes.emplace_back(Ty);
    }
    SmallVector<Type> ResultTypes;
    SmallVector<int64_t> WriteBacks;
    for (auto *CT : OTs) {
      auto Ty = getTensorTypeOfShape(CT->getShape());
      ResultTypes.emplace_back(Ty);
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
                         OB.getI64IntegerAttr(CF.getNumInputSizs())),
         OB.getNamedAttr("wirtebacks", OB.getI64ArrayAttr(WriteBacks))});
    auto F = OB.create<func::FuncOp>(Loc, CF.getName(), FT, FuncAttr);
    auto EntryB = F.addEntryBlock();
    OB.setInsertionPointToEnd(EntryB);
    // Map input tensor to mlir value
    for (size_t i = 0; i < ITs.size(); i++) {
      SymbolValueMap[ITs[i]->getName()] = EntryB->getArgument(i);
    }
    for (auto Index : CF.getArgIndices()) {
      SymbolValueMap[Index->getName()] =
          OB.create<arith::ConstantIndexOp>(Loc, 0);
    }
    for (auto Index : CF.getConstIndices()) {
      SymbolValueMap[Index->getName()] =
          OB.create<arith::ConstantIndexOp>(Loc, Index->getValue());
    }
    // Generate Contraction Dialect of trees
    auto &CTs = CF.getContractionTrees();
    for (auto &CT : CTs) {
      generateContractionTree(*CT);
    }
    // create mlir return
    // writebacks point to the argument tensor
    SmallVector<Value> ReturnValues;
    for (auto &CT : OTs) {
      ReturnValues.emplace_back(SymbolValueMap[CT->getName()]);
    }
    OB.create<func::ReturnOp>(Loc, TypeRange(), ReturnValues);
    return M;
  }

private:
  void generateContractionTree(ContractionTree &CT) {
    auto ValueIndices = generateNode(CT.getRootNode());
    SymbolValueMap[CT.getName()] = ValueIndices.first;
  }

  IndexedValueTy generateNode(Node &N) {
    if (isa<RootNode>(N)) {
      return generateNode(cast<RootNode>(N));
    } else if (isa<ContractionNode>(N)) {
      return generateNode(cast<ContractionNode>(N));
    } else if (isa<AdditionNode>(N)) {
      return generateNode<AddOp>(cast<ArithNode>(N));
    } else if (isa<MultiplicationNode>(N)) {
      return generateNode<MulOp>(cast<ArithNode>(N));
    } else if (isa<LeafNode>(N)) {
      return generateNode(cast<LeafNode>(N));
    } else if (isa<ConstantNode>(N)) {
      return generateNode(cast<ConstantNode>(N));
    } else if (isa<NegativeNode>(N)) {
      return generateNode(cast<NegativeNode>(N));
    } else {
      errs() << "Generate Node of Kind: " << N.getKind() << "\n";
      llvm_unreachable("Undefined Node Kind\n");
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
  IndexedValueTy generateNode(RootNode &RN) {
    auto ValueIndices = generateNode(RN.getChild());
    Value V = ValueIndices.first;
    auto &IndexUses = ValueIndices.second;
    auto &IndexDefs = RN.getIndexes();
    assert(IndexUses.size() == IndexDefs.size());
    SmallVector<int64_t> AxesMap;
    for (auto Idx : IndexUses) {
      for (size_t Axis = 0; Axis < IndexDefs.size(); Axis++) {
        if (Idx == IndexDefs[Axis].get()) {
          AxesMap.emplace_back(Axis);
          break;
        }
      }
    }
    assert(IndexUses.size() == AxesMap.size());
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
    return {V, {}};
  }
  IndexedValueTy generateNode(ContractionNode &CN) {
    auto ValueIndices = generateNode(CN.getChild());
    Value V = ValueIndices.first;
    auto InSizes = getTensorSizes(V);
    auto &IndexUses = ValueIndices.second;
    auto &IndexDefs = CN.getIndexes();
    IndexUsesTy OutIndexes;
    SmallVector<int64_t> OutSizes;
    SmallVector<int64_t> ContractedAxes;
    for (size_t Axis = 0; Axis < IndexUses.size(); Axis++) {
      auto &Idx1 = IndexUses[Axis];
      if (llvm::none_of(IndexDefs,
                        [&](auto &Idx2) { return Idx1 == Idx2.get(); })) {
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
    return {V, OutIndexes};
  }
  template <class T> IndexedValueTy generateNode(ArithNode &AN) {
    auto &Childs = AN.getChilds();
    SmallVector<Value> Operands;
    SmallVector<int64_t> OutSizes;
    IndexUsesTy OutIndexes;
    SmallVector<SmallVector<int64_t>> OutAxesMaps(Childs.size());
    for (size_t Ci = 0; Ci < Childs.size(); Ci++) {
      auto &C = Childs[Ci];
      auto ValueIndices = generateNode(*C);
      Value V = ValueIndices.first;
      Operands.emplace_back(V);
      auto Sizes = getTensorSizes(V);
      auto &Indexes = ValueIndices.second;
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
    return {V, OutIndexes};
  }
  IndexedValueTy generateNode(LeafNode &LN) {
    StringRef Name;
    if (llvm::isa<TreeAsLeafNode>(LN)) {
      Name = llvm::cast<TreeAsLeafNode>(LN).getTree()->getName();
    } else if (llvm::isa<InputAsLeafNode>(LN)) {
      Name = llvm::cast<InputAsLeafNode>(LN).getInput()->getName();
    }
    Value V = SymbolValueMap[Name];
    IndexUsesTy IIndexUses;
    // U unique
    IndexUsesTy UIndexUses;
    {
      auto &IndexUses = LN.getIndexUses();
      ShapeTy SliceShape;
      SmallVector<int64_t> SliceAxes;
      SmallVector<Value> PIndexValues;
      for (size_t IU = 0; IU < IndexUses.size(); ++IU) {
        auto Index = IndexUses[IU];
        if (auto PIndex = llvm::dyn_cast<ParamIndexSymbol>(Index)) {
          SliceAxes.emplace_back(IU);
          PIndexValues.emplace_back(SymbolValueMap[PIndex->getName()]);
        } else if (auto IIndex = llvm::dyn_cast<IterIndexSymbol>(Index)) {
          IIndexUses.emplace_back(IIndex);
          SliceShape.emplace_back(IIndex->getSize());
        } else {
          llvm_unreachable("");
        }
      }
      if (SliceAxes.size()) {
        auto SliceType = getTensorTypeOfShape(SliceShape);
        auto SliceAttr = OB.getI64ArrayAttr(SliceAxes);
        V = OB.create<SliceOp>(Loc, SliceType, V, PIndexValues, SliceAttr);
      }
    }
    {
      DenseMap<IndexPtr, SmallVector<int64_t>> IndexPosMap;
      ShapeTy DiagShape;
      for (size_t IU = 0; IU < IIndexUses.size(); ++IU) {
        auto Index = IIndexUses[IU];
        if (!IndexPosMap.count(Index)) {
          UIndexUses.emplace_back(Index);
          DiagShape.emplace_back(Index->getSize());
        }
        IndexPosMap[Index].emplace_back(IU);
      }
      SmallVector<Attribute> AxesPairs;
      for (auto Index : UIndexUses) {
        if (IndexPosMap[Index].size() >= 2) {
          AxesPairs.emplace_back(OB.getI64ArrayAttr(IndexPosMap[Index]));
        }
      }
      if (AxesPairs.size()) {
        auto DiagType = getTensorTypeOfShape(DiagShape);
        auto AxesPairsAttr = OB.getArrayAttr(AxesPairs);
        V = OB.create<DiagonalOp>(Loc, DiagType, V, AxesPairsAttr);
      }
    }
    return {V, UIndexUses};
  }
  IndexedValueTy generateNode(ConstantNode &CN) {
    auto Attr = OB.getF64FloatAttr(CN.getValue());
    auto Encoding = OB.getI64ArrayAttr({});
    auto ResultType = RankedTensorType::get({}, OB.getF64Type(), Encoding);
    Value V = OB.create<ConstantOp>(Loc, ResultType, Attr);
    return {V, {}};
  }
  IndexedValueTy generateNode(NegativeNode &NN) {
    auto ValueIndices = generateNode(NN.getChild());
    Value V = ValueIndices.first;
    auto &IndexUses = ValueIndices.second;
    V = OB.create<NegOp>(Loc, V.getType(), V);
    return {V, IndexUses};
  }
};

OwningOpRef<ModuleOp> MLIRGen(MLIRContext &Ctx, ContractionForest &CF) {
  return MLIRGenImpl(Ctx, CF).MLIRGen();
}
} // namespace contraction
} // namespace mlir
