#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Verifier.h"

#include "TCC/Dialect/TC/IR/TCOps.h"
#include "TCC/MLIRGen.h"

using llvm::cast;
using llvm::isa;

using llvm::errs;

using namespace tc;

namespace mlir {
namespace tc {
class MLIRGenImpl {
private:
  ContractionForest &CF;
  ModuleOp M;
  OpBuilder builder;
  Location Loc;

public:
  MLIRGenImpl(MLIRContext &Ctx, ContractionForest &CF)
      : CF(CF), builder(&Ctx), Loc(UnknownLoc::get(&Ctx)){};

private:
  llvm::DenseMap<StringRef, Value> SymbolValueMap;

  using IndexedValueTy = std::pair<Value, IndexUsesTy>;

  auto getTensorTypeOfShape(SizesTy &Shape) {
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
    auto EncodingAttr = builder.getI64ArrayAttr(Encoding);
    return RankedTensorType::get(TShape, builder.getF64Type(), EncodingAttr);
  }

public:
  ModuleOp MLIRGen() {
    // create mlir module
    M = ModuleOp::create(Loc);
    builder.setInsertionPointToEnd(M.getBody());
    // create mlir function
    auto &ArgSizes = CF.getArgSizes();
    auto &ArgIndices = CF.getArgIndices();
    auto &ArgTensors = CF.getArgTensors();
    auto &RetTensors = CF.getRetTensors();
    SmallVector<Type> InputTypes;
    InputTypes.append(
        SmallVector<Type>(ArgSizes.size(), builder.getIndexType()));
    InputTypes.append(
        SmallVector<Type>(ArgIndices.size(), builder.getIndexType()));
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
      if (ITIdx >= 0)
        WriteBacks.emplace_back(ITIdx + ArgSizes.size() + ArgIndices.size());
      else
        WriteBacks.emplace_back(-1);
    }
    auto FuncType = builder.getFunctionType(InputTypes, ResultTypes);
    auto FuncAttr = SmallVector<NamedAttribute>(
        {builder.getNamedAttr("num_input_sizes",
                              builder.getI64IntegerAttr(ArgSizes.size())),
         builder.getNamedAttr("num_input_indices",
                              builder.getI64IntegerAttr(ArgIndices.size())),
         builder.getNamedAttr("num_input_tensors",
                              builder.getI64IntegerAttr(ArgTensors.size())),
         builder.getNamedAttr("wirtebacks",
                              builder.getI64ArrayAttr(WriteBacks))});
    auto Func =
        builder.create<func::FuncOp>(Loc, CF.getName(), FuncType, FuncAttr);
    auto EntryB = Func.addEntryBlock();
    builder.setInsertionPointToEnd(EntryB);
    // Set argument names
    auto argIdxOffset = ArgSizes.size() + ArgIndices.size();
    for (auto [argIdx, IT] : llvm::enumerate(ArgTensors)) {
      argIdx = argIdx + argIdxOffset;
      Func.setArgAttr(argIdx, "func.name",
                      builder.getStringAttr(IT->getName()));
    }
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
          builder.create<arith::ConstantIndexOp>(Loc, Index->getValue());
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
    builder.create<func::ReturnOp>(Loc, TypeRange(), ReturnValues);
    return M;
  }

private:
  void generateContractionTree(ContractionTree &CT) {
    auto [value, _] = generateNode(CT.getRootNode());
    SymbolValueMap[CT.getName()] = value;
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
    auto [input, indexUses] = generateNode(node.getChild());
    auto &indexDefs = node.getIndexes();
    assert(indexUses.size() == indexDefs.size());
    AxesMapTy axesMap;
    for (auto index : indexUses) {
      for (auto axis : llvm::seq<int64_t>(0, indexDefs.size())) {
        if (index == indexDefs[axis].get()) {
          axesMap.emplace_back(axis);
          break;
        }
      }
    }
    assert(indexUses.size() == axesMap.size());
    SizesTy shape;
    for (auto &index : indexDefs) {
      shape.emplace_back(index->getSize());
    }
    auto outputType = getTensorTypeOfShape(shape);
    auto value = builder.create<PermuteOp>(Loc, input, axesMap);
    assert(outputType == value.getType());
    return {value, {}};
  }

  // 12-21
  IndexedValueTy generateNode(ContractionNode &node) {
    auto [input, indexUses] = generateNode(node.getChild());
    auto &indexDefs = node.getIndexes();
    SizesTy outputShape;
    IndexUsesTy outputIndexUses;
    AxesTy axes;
    for (auto axis : llvm::seq<int64_t>(0, indexUses.size())) {
      auto index1 = indexUses[axis];
      if (llvm::none_of(indexDefs,
                        [&](auto &index2) { return index1 == index2.get(); })) {
        outputShape.emplace_back(index1->getSize());
        outputIndexUses.emplace_back(index1);
      } else {
        axes.emplace_back(axis);
      }
    }
    auto outputType = getTensorTypeOfShape(outputShape);
    Value value = builder.create<ContractOp>(Loc, input, axes);
    assert(outputType == value.getType());
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
    SizesTy outputShape;
    IndexUsesTy outputIndexUses;
    AxesMapsTy axesMaps(childs.size());
    for (auto [child, axesMap] : llvm::zip(childs, axesMaps)) {
      auto [value, indexUses] = generateNode(*child);
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
    Value value =
        builder.create<AddOp>(Loc, operands, axesMaps, node.getAddOrSub());
    assert(outputType == value.getType());
    return {value, outputIndexUses};
  }

  IndexedValueTy generateNode(MultiplicationNode &node) {
    auto &childs = node.getChilds();
    SmallVector<Value> operands;
    SizesTy outputShape;
    IndexUsesTy outputIndexUses;
    AxesMapsTy axesMaps(childs.size());
    for (auto [child, axesMap] : llvm::zip(childs, axesMaps)) {
      auto [value, indexUses] = generateNode(*child);
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
    Value value = builder.create<MulOp>(Loc, operands, axesMaps);
    assert(outputType == value.getType());
    return {value, outputIndexUses};
  }

  IndexedValueTy generateNode(LeafNode &LN) {
    StringRef Name;
    if (llvm::isa<TreeTensorNode>(LN)) {
      Name = llvm::cast<TreeTensorNode>(LN).getTree()->getName();
    } else if (llvm::isa<InputTensorNode>(LN)) {
      Name = llvm::cast<InputTensorNode>(LN).getInput()->getName();
    }
    Value value = SymbolValueMap[Name];
    assert(value);
    IndexUsesTy IIndexUses;
    // U unique
    IndexUsesTy UIndexUses;
    {
      auto &IndexUses = LN.getIndexUses();
      SizesTy SliceShape;
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
        value = builder.create<SliceOp>(Loc, value, PIndexValues, SliceAxes);
        assert(SliceType == value.getType());
      }
    }
    {
      DenseMap<IndexPtr, SmallVector<int64_t>> IndexPosMap;
      SizesTy DiagShape;
      for (size_t IU = 0; IU < IIndexUses.size(); ++IU) {
        auto Index = IIndexUses[IU];
        if (!IndexPosMap.count(Index)) {
          UIndexUses.emplace_back(Index);
          DiagShape.emplace_back(Index->getSize());
        }
        IndexPosMap[Index].emplace_back(IU);
      }
      AxesPairsTy AxesPairs;
      for (auto Index : UIndexUses) {
        // if (IndexPosMap[Index].size() >= 2) {
        AxesPairs.emplace_back(IndexPosMap[Index]);
        // }
      }
      if (UIndexUses.size() < IIndexUses.size()) {
        auto DiagType = getTensorTypeOfShape(DiagShape);
        value = builder.create<DiagonalOp>(Loc, value, AxesPairs);
        assert(DiagType == value.getType());
      }
    }
    return {value, UIndexUses};
  }

  IndexedValueTy generateNode(ConstantNode &node) {
    auto value = builder.create<ConstantOp>(Loc, node.getValue());
    return {value, {}};
  }

  IndexedValueTy generateNode(NegativeNode &node) {
    auto [input, indexUses] = generateNode(node.getChild());
    Value value = builder.create<NegOp>(Loc, input);
    return {value, indexUses};
  }
};

OwningOpRef<ModuleOp> MLIRGen(MLIRContext &Ctx, ContractionForest &CF) {
  return MLIRGenImpl(Ctx, CF).MLIRGen();
}
} // namespace tc
} // namespace mlir
