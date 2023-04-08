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
#include "TCC/Dialect/Utils/FuncUtils.h"
#include "TCC/MLIRGen.h"

using llvm::cast;
using llvm::isa;

using llvm::errs;

#define DEBUG_TYPE "tcc-mlirgen"

using namespace tc;

namespace mlir {
namespace tc {
class MLIRGenImpl {
private:
  ContractionForest &CF;
  ModuleOp moduleOp;
  OpBuilder builder;
  Location loc;

public:
  MLIRGenImpl(MLIRContext &Ctx, ContractionForest &CF)
      : CF(CF), builder(&Ctx), loc(UnknownLoc::get(&Ctx)){};

private:
  llvm::DenseMap<StringRef, Value> nameValueMap;

  using IndexedValueTy = std::pair<Value, IndexUsesTy>;

  auto getTensorTypeOfShape(SizesTy &symbolShape) {
    auto shape = llvm::to_vector(
        llvm::map_range(symbolShape, [](SizePtr size) -> int64_t {
          if (size->isConstant())
            return size->getUpperBound();
          return -1;
        }));
    auto sizeIDs = llvm::to_vector(llvm::map_range(
        symbolShape, [](SizePtr size) -> int64_t { return size->getUID(); }));
    return mlir::tc::getEncodedTensorType(builder, shape, sizeIDs);
  }

public:
  ModuleOp MLIRGen() {
    // Create mlir module
    moduleOp = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(moduleOp.getBody());
    // Get function related data from CF
    auto &argSizes = CF.getArgSizes();
    auto &argIndices = CF.getArgIndices();
    auto &argTensors = CF.getArgTensors();
    auto &constantSizes = CF.getConstantSizes();
    auto &constantIndices = CF.getConstantIndices();
    auto &resultTensors = CF.getResultTensors();
    // Function type
    SmallVector<Type> inputTypes;
    SmallVector<Type> resultTypes;
    auto indexType = builder.getIndexType();
    inputTypes.append(argSizes.size(), indexType);
    inputTypes.append(argIndices.size(), indexType);
    for (auto &inputTensor : argTensors) {
      auto RTT = getTensorTypeOfShape(inputTensor->getShape());
      inputTypes.emplace_back(RTT);
    }
    for (auto resultTensor : resultTensors) {
      auto RTT = getTensorTypeOfShape(resultTensor->getShape());
      resultTypes.emplace_back(RTT);
    }
    auto funcType = builder.getFunctionType(inputTypes, resultTypes);
    // Function attributes
    auto numSizesAttr = builder.getI64IntegerAttr(argSizes.size());
    auto numIndicesAttr = builder.getI64IntegerAttr(argIndices.size());
    auto numInputTensorsAttr = builder.getI64IntegerAttr(argTensors.size());
    SmallVector<int64_t> sizeLBs, sizeUBs;
    for (auto size : argSizes) {
      sizeLBs.emplace_back(size->getLowerBound());
      sizeUBs.emplace_back(size->getUpperBound());
    }
    for (auto size : constantSizes) {
      sizeLBs.emplace_back(size->getLowerBound());
      sizeUBs.emplace_back(size->getUpperBound());
    }
    auto sizeLBsAttr = builder.getDenseI64ArrayAttr(sizeLBs);
    auto sizeUBsAttr = builder.getDenseI64ArrayAttr(sizeUBs);
    SmallVector<int64_t> writeBacks;
    for (auto resutlTensor : resultTensors) {
      auto name = resutlTensor->getName();
      auto inputIdx = -1;
      for (auto [idx, inputTensor] : llvm::enumerate(argTensors)) {
        if (name == inputTensor->getName()) {
          inputIdx = idx;
          break;
        }
      }
      writeBacks.emplace_back(inputIdx);
    }
    auto writeBackAttr = builder.getI64ArrayAttr(writeBacks);
    auto funcAttr = SmallVector<NamedAttribute>(
        {// builder.getNamedAttr("num_input_sizes", numSizesAttr),
         builder.getNamedAttr("num_input_indices", numIndicesAttr),
         builder.getNamedAttr("num_input_tensors", numInputTensorsAttr),
         builder.getNamedAttr("sizes_lower_bound", sizeLBsAttr),
         builder.getNamedAttr("sizes_upper_bound", sizeUBsAttr),
         builder.getNamedAttr("wirtebacks", writeBackAttr)});
    auto funcOp =
        builder.create<func::FuncOp>(loc, CF.getName(), funcType, funcAttr);
    utils::setNumInputSizes(funcOp, numSizesAttr);
    // Set argument attributes
    size_t offset;
    // sizes
    offset = 0;
    for (auto [argIdx, size] : llvm::enumerate(argSizes)) {
      auto argNameAttr = builder.getStringAttr(size->getUName());
      funcOp.setArgAttr(argIdx + offset, "func.name", argNameAttr);
    }
    offset += argSizes.size();
    // indices
    for (auto [argIdx, index] : llvm::enumerate(argIndices)) {
      auto argNameAttr = builder.getStringAttr(index->getName());
      funcOp.setArgAttr(argIdx + offset, "func.name", argNameAttr);
    }
    offset += argIndices.size();
    // tensors
    for (auto [argIdx, inputTensor] : llvm::enumerate(argTensors)) {
      auto argNameAttr = builder.getStringAttr(inputTensor->getName());
      funcOp.setArgAttr(argIdx + offset, "func.name", argNameAttr);
    }
    // Set result attributes
    for (auto [resultIdx, resultTensor] : llvm::enumerate(resultTensors)) {
      auto resultNameAttr = builder.getStringAttr(resultTensor->getName());
      funcOp.setResultAttr(resultIdx, "func.name", resultNameAttr);
    }

    auto entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToEnd(entryBlock);
    // Map input tensor to mlir value
    auto arguments = entryBlock->getArguments();
    offset = 0;
    for (auto [argIdx, size] : llvm::enumerate(argSizes)) {
      nameValueMap[size->getUName()] = arguments[argIdx + offset];
    }
    offset += argSizes.size();
    for (auto [argIdx, index] : llvm::enumerate(argIndices)) {
      nameValueMap[index->getName()] = arguments[argIdx + offset];
    }
    offset += argIndices.size();
    for (auto [argIdx, inputTensor] : llvm::enumerate(argTensors)) {
      nameValueMap[inputTensor->getName()] = arguments[argIdx + offset];
    }
    for (auto index : constantIndices) {
      auto constantOp =
          builder.create<arith::ConstantIndexOp>(loc, index->getValue());
      nameValueMap[index->getName()] = constantOp.getResult();
    }
    // Create calculations
    auto &treeTensors = CF.getContractionTrees();
    for (auto &treeTensor : treeTensors) {
      generateContractionTree(*treeTensor);
    }
    // Create mlir return
    SmallVector<Value> returnValues;
    for (auto resultTensor : resultTensors)
      returnValues.emplace_back(nameValueMap[resultTensor->getName()]);
    builder.create<ReturnOp>(loc, returnValues);
    return moduleOp;
  }

private:
  void generateContractionTree(ContractionTree &CT) {
    auto [value, _] = generateNode(CT.getRootNode());
    nameValueMap[CT.getName()] = value;
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
    auto value = builder.create<PermuteOp>(loc, input, axesMap);
    assert(checkResultType(getTensorTypeOfShape(shape), value.getType()));
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
    Value value = builder.create<ContractOp>(loc, input, axes);
    assert(checkResultType(getTensorTypeOfShape(outputShape), value.getType()));
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
    Value value =
        builder.create<AddOp>(loc, operands, axesMaps, node.getAddOrSub());
    assert(checkResultType(getTensorTypeOfShape(outputShape), value.getType()));
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
    Value value = builder.create<MulOp>(loc, operands, axesMaps);
    assert(checkResultType(getTensorTypeOfShape(outputShape), value.getType()));
    return {value, outputIndexUses};
  }

  IndexedValueTy generateNode(LeafNode &LN) {
    StringRef Name;
    if (llvm::isa<TreeTensorNode>(LN)) {
      Name = llvm::cast<TreeTensorNode>(LN).getTree()->getName();
    } else if (llvm::isa<InputTensorNode>(LN)) {
      Name = llvm::cast<InputTensorNode>(LN).getInput()->getName();
    }
    Value value = nameValueMap[Name];
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
          PIndexValues.emplace_back(nameValueMap[PIndex->getName()]);
        } else if (auto IIndex = llvm::dyn_cast<IterIndexSymbol>(Index)) {
          IIndexUses.emplace_back(IIndex);
          SliceShape.emplace_back(IIndex->getSize());
        } else {
          llvm_unreachable("");
        }
      }
      if (SliceAxes.size()) {
        auto SliceType = getTensorTypeOfShape(SliceShape);
        value = builder.create<SliceOp>(loc, value, PIndexValues, SliceAxes);
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
        value = builder.create<DiagonalOp>(loc, value, AxesPairs);
        assert(DiagType == value.getType());
      }
    }
    return {value, UIndexUses};
  }

  IndexedValueTy generateNode(ConstantNode &node) {
    auto value = builder.create<ConstantOp>(loc, node.getValue());
    return {value, {}};
  }

  IndexedValueTy generateNode(NegativeNode &node) {
    auto [input, indexUses] = generateNode(node.getChild());
    Value value = builder.create<NegOp>(loc, input);
    return {value, indexUses};
  }

  static bool checkResultType(Type type1, Type type2) {
    if (type1 != type2) {
      auto RTT1 = type1.cast<RankedTensorType>();
      auto RTT2 = type2.cast<RankedTensorType>();
      if (RTT1.getShape() != RTT2.getShape() ||
          mlir::tc::getTensorSizeIDs(RTT1) != mlir::tc::getTensorSizeIDs(RTT2))
        return false;
    }
    return true;
  }
};

OwningOpRef<ModuleOp> MLIRGen(MLIRContext &Ctx, ContractionForest &CF) {
  return MLIRGenImpl(Ctx, CF).MLIRGen();
}
} // namespace tc
} // namespace mlir
