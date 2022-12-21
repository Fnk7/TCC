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
    auto &ArgSizes = CF.getArgSizes();
    auto &ArgIndices = CF.getArgIndices();
    auto &ArgTensors = CF.getArgTensors();
    auto &RetTensors = CF.getRetTensors();
    SmallVector<Type> InputTypes;
    InputTypes.append(SmallVector<Type>(ArgSizes.size(), OB.getIndexType()));
    InputTypes.append(SmallVector<Type>(ArgIndices.size(), OB.getIndexType()));
    for (auto &IT : ArgTensors) {
      auto Ty = getTensorTypeOfShape(IT->getShape());
      InputTypes.emplace_back(Ty);
    }
    SmallVector<Type> ResultTypes;
    SmallVector<int64_t> WriteBacks;
    for (auto *CT : RetTensors) {
      auto Ty = getTensorTypeOfShape(CT->getShape());
      ResultTypes.emplace_back(Ty);
      int64_t ITIdx = ArgTensors.size() - 1;
      for (; ITIdx >= 0; ITIdx--) {
        if (CT->getName() == ArgTensors[ITIdx]->getName())
          break;
      }
      WriteBacks.emplace_back(ITIdx);
    }
    auto FuncType = OB.getFunctionType(InputTypes, ResultTypes);
    auto FuncAttr = SmallVector<NamedAttribute>(
        {OB.getNamedAttr("num_input_sizes",
                         OB.getI64IntegerAttr(ArgSizes.size())),
         OB.getNamedAttr("num_input_indices",
                         OB.getI64IntegerAttr(ArgIndices.size())),
         OB.getNamedAttr("num_input_tensors",
                         OB.getI64IntegerAttr(ArgTensors.size())),
         OB.getNamedAttr("wirtebacks", OB.getI64ArrayAttr(WriteBacks))});
    auto Func = OB.create<func::FuncOp>(Loc, CF.getName(), FuncType, FuncAttr);
    auto EntryB = Func.addEntryBlock();
    OB.setInsertionPointToEnd(EntryB);
    // Map input tensor to mlir value
    size_t Offset = 0;
    for (size_t ArgIdx = 0; ArgIdx < ArgSizes.size(); ArgIdx++) {
      SymbolValueMap[ArgSizes[ArgIdx]->getUName()] =
          EntryB->getArgument(ArgIdx + Offset);
    }
    Offset += ArgSizes.size();
    for (size_t ArgIdx = 0; ArgIdx < ArgIndices.size(); ArgIdx++) {
      SymbolValueMap[ArgIndices[ArgIdx]->getName()] =
          EntryB->getArgument(ArgIdx + Offset);
    }
    Offset += ArgIndices.size();
    for (size_t ArgIdx = 0; ArgIdx < ArgTensors.size(); ArgIdx++) {
      SymbolValueMap[ArgTensors[ArgIdx]->getName()] =
          EntryB->getArgument(ArgIdx + Offset);
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
    for (auto &CT : RetTensors) {
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
      return generateNode(cast<AdditionNode>(N));
    } else if (isa<MultiplicationNode>(N)) {
      return generateNode(cast<MultiplicationNode>(N));
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

  // 12-21
  IndexedValueTy generateNode(RootNode &node) {
    auto valueIndices = generateNode(node.getChild());
    auto value = valueIndices.first;
    auto &indexUses = valueIndices.second;
    auto &indexDefs = node.getIndexes();
    assert(indexUses.size() == indexDefs.size());
    SmallVector<int64_t> axesMap;
    for (auto index : indexUses) {
      for (auto axis : llvm::seq<int64_t>(0, indexDefs.size())) {
        if (index == indexDefs[axis].get()) {
          axesMap.emplace_back(axis);
          break;
        }
      }
    }
    assert(indexUses.size() == axesMap.size());
    ShapeTy shape;
    for (auto &index : indexDefs) {
      shape.emplace_back(index->getSize());
    }
    auto outputType = getTensorTypeOfShape(shape);
    auto axesMapAttr = OB.getI64ArrayAttr(axesMap);
    value = OB.create<PermuteOp>(Loc, outputType, value, axesMapAttr);
    return {value, {}};
  }

  // 12-21
  IndexedValueTy generateNode(ContractionNode &node) {
    auto valueIndices = generateNode(node.getChild());
    auto value = valueIndices.first;
    auto &indexUses = valueIndices.second;
    auto &indexDefs = node.getIndexes();
    ShapeTy outputShape;
    IndexUsesTy outputIndexUses;
    SmallVector<int64_t> contractedAxes;
    for (auto axis : llvm::seq<int64_t>(0, indexUses.size())) {
      auto index1 = indexUses[axis];
      if (llvm::none_of(indexDefs,
                        [&](auto &index2) { return index1 == index2.get(); })) {
        outputShape.emplace_back(index1->getSize());
        outputIndexUses.emplace_back(index1);
      } else {
        contractedAxes.emplace_back(axis);
      }
    }
    auto outputType = getTensorTypeOfShape(outputShape);
    auto contractedAxesAtrr = OB.getI64ArrayAttr(contractedAxes);
    value = OB.create<ContractOp>(Loc, outputType, value, contractedAxesAtrr);
    return {value, outputIndexUses};
  }

  static auto getIndexListArrayAttribute(OpBuilder &builder,
                                         ArrayRef<SmallVector<int64_t>> list) {
    SmallVector<Attribute, 4> array;
    for (auto item : list) {
      array.emplace_back(builder.getI64ArrayAttr(item));
    }
    return builder.getArrayAttr(array);
  }

  // 12-21
  IndexedValueTy generateNode(AdditionNode &node) {
    auto &childs = node.getChilds();
    SmallVector<Value> operands;
    ShapeTy outputShape;
    IndexUsesTy outputIndexUses;
    SmallVector<SmallVector<int64_t>> axesMaps(childs.size());
    for (auto [child, axesMap] : llvm::zip(childs, axesMaps)) {
      auto valueIndices = generateNode(*child);
      auto value = valueIndices.first;
      auto indexUses = valueIndices.second;
      operands.emplace_back(value);
      for (auto index : indexUses) {
        size_t axis = 0;
        for (; axis < outputIndexUses.size(); axis++) {
          if (index == outputIndexUses[axis])
            break;
        }
        if (axis == outputIndexUses.size()) {
          outputShape.emplace_back(index->getSize());
          outputIndexUses.emplace_back(index);
        }
        axesMap.emplace_back(axis);
      }
    }
    auto outputType = getTensorTypeOfShape(outputShape);
    auto axesMapsAttr = getIndexListArrayAttribute(OB, axesMaps);
    auto signsAttr = OB.getBoolArrayAttr(node.getAddOrSub());
    Value value =
        OB.create<AddOp>(Loc, outputType, operands, axesMapsAttr, signsAttr);
    return {value, outputIndexUses};
  }

  IndexedValueTy generateNode(MultiplicationNode &node) {
    auto &childs = node.getChilds();
    SmallVector<Value> operands;
    ShapeTy outputShape;
    IndexUsesTy outputIndexUses;
    SmallVector<SmallVector<int64_t>> axesMaps(childs.size());
    for (auto [child, axesMap] : llvm::zip(childs, axesMaps)) {
      auto valueIndices = generateNode(*child);
      auto value = valueIndices.first;
      auto indexUses = valueIndices.second;
      operands.emplace_back(value);
      for (auto index : indexUses) {
        size_t axis = 0;
        for (; axis < outputIndexUses.size(); axis++) {
          if (index == outputIndexUses[axis])
            break;
        }
        if (axis == outputIndexUses.size()) {
          outputShape.emplace_back(index->getSize());
          outputIndexUses.emplace_back(index);
        }
        axesMap.emplace_back(axis);
      }
    }
    auto outputType = getTensorTypeOfShape(outputShape);
    auto axesMapsAttr = getIndexListArrayAttribute(OB, axesMaps);
    Value value = OB.create<MulOp>(Loc, outputType, operands, axesMapsAttr);
    return {value, outputIndexUses};
  }

  IndexedValueTy generateNode(LeafNode &LN) {
    StringRef Name;
    if (llvm::isa<TreeAsLeafNode>(LN)) {
      Name = llvm::cast<TreeAsLeafNode>(LN).getTree()->getName();
    } else if (llvm::isa<InputAsLeafNode>(LN)) {
      Name = llvm::cast<InputAsLeafNode>(LN).getInput()->getName();
    }
    Value V = SymbolValueMap[Name];
    assert(V);
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
        // if (IndexPosMap[Index].size() >= 2) {
        AxesPairs.emplace_back(OB.getI64ArrayAttr(IndexPosMap[Index]));
        // }
      }
      if (UIndexUses.size() < IIndexUses.size()) {
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
