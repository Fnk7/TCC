#include "llvm/ADT/Sequence.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "TCC/Dialect/TC/IR/TCOps.h"
#include "TCC/Passes.h"

namespace mlir {
namespace tc {
static SmallVector<int64_t> unpackIntegerArrayAttr(ArrayAttr Attr) {
  SmallVector<int64_t> Sizes;
  for (auto &I64 : Attr) {
    Sizes.emplace_back(I64.cast<IntegerAttr>().getValue().getSExtValue());
  }
  return Sizes;
}
static SmallVector<int64_t> getTensorSizeIDs(RankedTensorType RTT) {
  auto Attr = RTT.getEncoding().cast<ArrayAttr>();
  return unpackIntegerArrayAttr(Attr);
}
static SmallVector<int64_t> getTensorSizes(Value V) {
  auto RTT = V.getType().cast<RankedTensorType>();
  return getTensorSizeIDs(RTT);
}

static std::tuple<Value, SmallVector<Value>>
createBufferForTensor(OpBuilder &builder, Value tensorValue) {
  auto loc = tensorValue.getLoc();
  auto tensorType = cast<RankedTensorType>(tensorValue.getType());
  auto memrefType =
      MemRefType::get(tensorType.getShape(), tensorType.getElementType());
  auto shape = tensorType.getShape();
  auto sizeIDs = getTensorSizeIDs(tensorType);
  assert(shape.size() == sizeIDs.size());
  auto block = tensorValue.getParentBlock();
  SmallVector<Value> allocSizes;
  SmallVector<Value> sizeValues;
  for (auto [size, sizeID] : llvm::zip(shape, sizeIDs)) {
    if (size > 0) {
      auto sizeValue = builder.create<arith::ConstantIndexOp>(loc, size);
      sizeValues.emplace_back(sizeValue);
    } else {
      sizeValues.emplace_back(block->getArgument(sizeID));
      allocSizes.emplace_back(sizeValues.back());
    }
  }
  bool isAlloca = true;
  for (auto U : tensorValue.getUsers()) {
    if (dyn_cast<func::ReturnOp>(U)) {
      isAlloca = false;
    }
  }
  if (isAlloca) {
    auto alloca = builder.create<memref::AllocaOp>(loc, memrefType, allocSizes);
    return {alloca, sizeValues};
  } else {
    auto alloc = builder.create<memref::AllocOp>(loc, memrefType, allocSizes);
    return {alloc, sizeValues};
  }
}

static auto createNestedAffineForOps(OpBuilder &builder, Location loc,
                                     ArrayRef<Value> sizeValues) {
  Value zeroIndexValue = builder.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value> inductionValues;
  for (auto sizeValue : sizeValues) {
    auto forOp = builder.create<AffineForOp>(
        loc, zeroIndexValue, builder.getSymbolIdentityMap(), sizeValue,
        builder.getSymbolIdentityMap(), 1);
    inductionValues.emplace_back(forOp.getInductionVar());
    builder.setInsertionPointToStart(forOp.getBody());
  }
  return inductionValues;
}

template <typename BinaryOp, typename LoweredBinaryOp>
struct BinaryOpLowering : public OpConversionPattern<BinaryOp> {
  using OpConversionPattern<BinaryOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BinaryOp op, typename BinaryOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto [buffer, sizeValues] = createBufferForTensor(rewriter, op.getResult());
    auto inductionValues = createNestedAffineForOps(rewriter, loc, sizeValues);
    SmallVector<Value> Loads;
    auto AxesMapsAttr = op.getAxesMaps();
    for (size_t Operand = 0; Operand < AxesMapsAttr.size(); Operand++) {
      Value LoadFrom = adaptor.getInputs()[Operand];
      SmallVector<Value> Indexes;
      auto AxesMapAttr = cast<ArrayAttr>(AxesMapsAttr[Operand]);
      auto AxesMap = unpackIntegerArrayAttr(AxesMapAttr);
      for (auto Axis : AxesMap) {
        Indexes.emplace_back(inductionValues[Axis]);
      }
      Value Loaded = rewriter.create<memref::LoadOp>(loc, LoadFrom, Indexes);
      Loads.emplace_back(Loaded);
    }
    Value ToStore = Loads[0];
    for (size_t Operand = 1; Operand < Loads.size(); Operand++) {
      ToStore = rewriter.create<LoweredBinaryOp>(loc, ToStore, Loads[Operand]);
    }
    rewriter.create<memref::StoreOp>(loc, ToStore, buffer, inductionValues);
    rewriter.replaceOp(op, buffer);
    return success();
  }
};

using AddOpLowering = BinaryOpLowering<AddOp, arith::AddFOp>;
using MulOpLowering = BinaryOpLowering<MulOp, arith::MulFOp>;

struct ReturnOpLowering : public OpConversionPattern<func::ReturnOp> {
  using OpConversionPattern<func::ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp op, func::ReturnOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto Loc = rewriter.getUnknownLoc();
    auto &B = *(op->getBlock());
    auto ReturnValues = adaptor.getOperands();
    auto FuncAdaptor = func::FuncOpAdaptor(op->getParentOfType<func::FuncOp>());
    auto FuncAttr = FuncAdaptor.getAttributes();
    auto WriteBackAttr = FuncAttr.get("wirtebacks").cast<ArrayAttr>();
    auto WriteBacks = unpackIntegerArrayAttr(WriteBackAttr);
    auto NISAttr = FuncAttr.get("num_input_sizes").cast<IntegerAttr>();
    auto NIIAttr = FuncAttr.get("num_input_indices").cast<IntegerAttr>();
    auto Offset = NISAttr.getValue().getSExtValue();
    Offset += NIIAttr.getValue().getSExtValue();
    assert(WriteBacks.size() == ReturnValues.size());
    SmallVector<Value> NewReturnValues;
    for (auto [WriteBackIdx, ReturnValue] :
         llvm::zip(WriteBacks, ReturnValues)) {
      if (WriteBackIdx < 0) {
        NewReturnValues.emplace_back(ReturnValue);
      } else {
        auto InputValue = B.getArgument(WriteBackIdx + Offset);
        // auto CopyTo = (*(InputValue.getUsers().begin()))->getResult(0);
        auto Copy =
            rewriter.create<memref::CopyOp>(Loc, ReturnValue, InputValue);
        Copy->moveBefore(&B.back());
        auto Dealloc = rewriter.create<memref::DeallocOp>(Loc, ReturnValue);
        Dealloc->moveBefore(&B.back());
      }
    }
    rewriter.updateRootInPlace(op, [&] { op->setOperands(NewReturnValues); });
    return success();
  }
};

struct ConstantOpLowering : public OpConversionPattern<ConstantOp> {
  using OpConversionPattern<ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp op, ConstantOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto memrefType =
        MemRefType::get(op.getType().getShape(), op.getType().getElementType());
    auto allocaOp = rewriter.create<memref::AllocaOp>(loc, memrefType,
                                                      SmallVector<Value>());
    auto constOp = rewriter.create<arith::ConstantFloatOp>(
        loc, adaptor.getValue(), rewriter.getF64Type());
    rewriter.create<memref::StoreOp>(loc, constOp, allocaOp);
    rewriter.replaceOp(op, allocaOp.getResult());
    return success();
  }
};

struct ReturnOpLowering2 : public OpConversionPattern<func::ReturnOp> {
  using OpConversionPattern<func::ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp op, func::ReturnOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.updateRootInPlace(op,
                               [&] { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};

struct FuncOpLowering2 : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = rewriter.getUnknownLoc();
    auto funcType = op.getFunctionType();
    TypeConverter::SignatureConversion signature(funcType.getNumInputs());
    for (const auto &argType : llvm::enumerate(funcType.getInputs())) {
      auto convertedType = typeConverter->convertType(argType.value());
      if (!convertedType)
        return failure();
      signature.addInputs(argType.index(), convertedType);
    }
    SmallVector<Type> resultTypes;
    auto WriteBackAttr =
        adaptor.getAttributes().get("wirtebacks").cast<ArrayAttr>();
    auto WriteBacks = unpackIntegerArrayAttr(WriteBackAttr);
    for (auto [resultType, writeBack] :
         llvm::zip(funcType.getResults(), WriteBacks)) {
      if (writeBack < 0) {
        auto convertedType = typeConverter->convertType(resultType);
        resultTypes.push_back(convertedType);
      }
    }
    auto newFuncType =
        rewriter.getFunctionType(signature.getConvertedTypes(), resultTypes);
    auto newFuncOp =
        rewriter.create<func::FuncOp>(loc, op.getName(), newFuncType);
    for (const auto &namedAttr : op->getAttrs()) {
      if (namedAttr.getName() == FunctionOpInterface::getTypeAttrName() ||
          namedAttr.getName() == SymbolTable::getSymbolAttrName())
        continue;
      newFuncOp->setAttr(namedAttr.getName(), namedAttr.getValue());
    }
    rewriter.inlineRegionBefore(op.getRegion(), newFuncOp.getRegion(),
                                newFuncOp.getRegion().end());
    if (failed(rewriter.convertRegionTypes(&newFuncOp.getRegion(),
                                           *this->getTypeConverter())))
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

struct FuncOpLowering : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto Loc = rewriter.getUnknownLoc();
    rewriter.startRootUpdate(op);
    auto InputTypes = op.getArgumentTypes();
    auto ResultTypes = op.getResultTypes();
    auto FuncAttr = adaptor.getAttributes();
    auto NISAttr = FuncAttr.get("num_input_sizes").cast<IntegerAttr>();
    auto NumInputSizes = NISAttr.getValue().getSExtValue();
    auto NIIAttr = FuncAttr.get("num_input_indices").cast<IntegerAttr>();
    auto NumInputIndices = NIIAttr.getValue().getSExtValue();
    auto WriteBackAttr = FuncAttr.get("wirtebacks").cast<ArrayAttr>();
    auto WriteBacks = unpackIntegerArrayAttr(WriteBackAttr);
    SmallVector<Type> NewInputTypes(NumInputSizes + NumInputIndices,
                                    rewriter.getIndexType());
    SmallVector<Type> NewResultTypes;
    auto &R = op.getRegion();
    rewriter.setInsertionPointToStart(&(R.front()));
    Value ZeroOffset = rewriter.create<arith::ConstantIndexOp>(Loc, 0);
    unsigned ArgIdx = NumInputSizes + NumInputIndices;
    // for (; ArgIdx < NumInputSizes; ArgIdx++) {
    //   R.insertArgument(ArgIdx, rewriter.getIndexType(), Loc);
    // }
    for (; ArgIdx < InputTypes.size(); ArgIdx++) {
      auto T = InputTypes[ArgIdx];
      // for (auto T : InputTypes) {
      assert(T.isa<RankedTensorType>());
      auto RTT = T.cast<RankedTensorType>();
      auto ViewOpType = MemRefType::get(RTT.getShape(), RTT.getElementType());
      auto Arg = R.getArgument(ArgIdx++);
      auto ArgType = MemRefType::get({(int64_t)-1}, rewriter.getI8Type());
      NewInputTypes.emplace_back(ArgType);
      Arg.setType(ArgType);
      auto Sizes = getTensorSizeIDs(RTT);
      SmallVector<Value> SizeValues;
      for (auto &Sz : Sizes) {
        SizeValues.emplace_back(R.getArgument(Sz));
      }
      auto View = rewriter.create<memref::ViewOp>(Loc, ViewOpType, Arg,
                                                  ZeroOffset, SizeValues);
      Arg.replaceAllUsesExcept(View, View);
    }
    for (auto [WriteBack, T] : llvm::zip(WriteBacks, ResultTypes)) {
      assert(T.isa<RankedTensorType>());
      if (WriteBack < 0) {
        auto TT = T.cast<RankedTensorType>();
        NewResultTypes.emplace_back(
            MemRefType::get(TT.getShape(), TT.getElementType()));
      }
    }
    auto NewFuncType = rewriter.getFunctionType(NewInputTypes, NewResultTypes);
    op.setType(NewFuncType);
    op.setFunctionTypeAttr(TypeAttr::get(NewFuncType));
    rewriter.finalizeRootUpdate(op);
    return success();
  }
};
struct PermuteOpLowering : public OpConversionPattern<PermuteOp> {
  using OpConversionPattern<PermuteOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PermuteOp op, PermuteOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto Loc = op.getLoc();
    auto [buffer, SizeValues] = createBufferForTensor(rewriter, op.getResult());
    auto InductionVars = createNestedAffineForOps(rewriter, Loc, SizeValues);
    Value LoadFrom = adaptor.getInput();
    SmallVector<Value> Indexes;
    auto AxesMapAttr = op.getAxesMap();
    auto AxesMap = unpackIntegerArrayAttr(AxesMapAttr);
    for (auto Axis : AxesMap) {
      Indexes.emplace_back(InductionVars[Axis]);
    }
    Value ToStore = rewriter.create<memref::LoadOp>(Loc, LoadFrom, Indexes);
    rewriter.create<memref::StoreOp>(Loc, ToStore, buffer, InductionVars);
    rewriter.replaceOp(op, buffer);
    return success();
  }
};

struct DiagonalOpLowering : public OpConversionPattern<DiagonalOp> {
  using OpConversionPattern<DiagonalOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DiagonalOp op, DiagonalOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto Loc = op.getLoc();
    auto [Alloc, SizeValues] = createBufferForTensor(rewriter, op.getResult());
    SmallVector<Value> InductionVars =
        createNestedAffineForOps(rewriter, Loc, SizeValues);
    Value LoadFrom = adaptor.getInput();
    SmallVector<Value> Indexes(
        op.getOperand().getType().cast<RankedTensorType>().getShape().size());
    auto AxesMapAttr = op.getAxesPairs();
    for (auto [IndexValue, AxesAttr] : llvm::zip(InductionVars, AxesMapAttr)) {
      auto Axes = unpackIntegerArrayAttr(AxesAttr.cast<ArrayAttr>());
      for (auto Axis : Axes) {
        Indexes[Axis] = IndexValue;
      }
    }
    Value ToStore = rewriter.create<memref::LoadOp>(Loc, LoadFrom, Indexes);
    rewriter.create<memref::StoreOp>(Loc, ToStore, Alloc, InductionVars);
    rewriter.replaceOp(op, Alloc);
    return success();
  }
};

struct SliceOpLowering : public OpConversionPattern<SliceOp> {
  using OpConversionPattern<SliceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SliceOp op, SliceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto Loc = op.getLoc();
    auto [Alloc, SizeValues] = createBufferForTensor(rewriter, op.getResult());
    SmallVector<Value> InductionVars =
        createNestedAffineForOps(rewriter, Loc, SizeValues);
    Value LoadFrom = adaptor.getInput();
    SmallVector<Value> LoadIndices(
        op.getOperand(0).getType().cast<RankedTensorType>().getShape().size());
    auto AxesAttr = op.getAxesAttr();
    for (auto [Axis, IndexValue] :
         llvm::zip(unpackIntegerArrayAttr(AxesAttr), op.getIndices())) {
      LoadIndices[Axis] = IndexValue;
    }
    unsigned IndexPos = 0;
    for (auto IndexValue : InductionVars) {
      while (LoadIndices[IndexPos]) {
        IndexPos++;
      }
      LoadIndices[IndexPos] = IndexValue;
    }
    Value ToStore = rewriter.create<memref::LoadOp>(Loc, LoadFrom, LoadIndices);
    rewriter.create<memref::StoreOp>(Loc, ToStore, Alloc, InductionVars);
    rewriter.replaceOp(op, Alloc);
    return success();
  }
};

struct NegOpLowering : public OpConversionPattern<NegOp> {
  using OpConversionPattern<NegOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(NegOp op, NegOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto Loc = op.getLoc();
    auto [alloc, sizeValues] = createBufferForTensor(rewriter, op.getResult());
    auto InductionVars = createNestedAffineForOps(rewriter, Loc, sizeValues);
    Value LoadFrom = adaptor.getInput();
    Value ToStore =
        rewriter.create<memref::LoadOp>(Loc, LoadFrom, InductionVars);
    ToStore = rewriter.create<arith::NegFOp>(Loc, ToStore);
    rewriter.create<memref::StoreOp>(Loc, ToStore, alloc, InductionVars);
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

struct ContractOpLowering : public OpConversionPattern<ContractOp> {
  using OpConversionPattern<ContractOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ContractOp op, ContractOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto Loc = op.getLoc();
    auto [Alloc, SizeValues] = createBufferForTensor(rewriter, op.getResult());
    auto InType = op.getResult().getType().cast<RankedTensorType>();
    auto &B = *(op->getBlock());
    Value ZeroIndexVar = rewriter.create<arith::ConstantIndexOp>(Loc, 0);
    SmallVector<Value> InductionVars =
        createNestedAffineForOps(rewriter, Loc, SizeValues);
    // Inner Reduction Loop
    auto Sizes2 = getTensorSizes(op.getOperand());
    auto AxesAttr = op.getAxesAttr();
    auto ContractedAxes = unpackIntegerArrayAttr(AxesAttr);
    SmallVector<Value> SizeValues2;
    for (auto CAxis : ContractedAxes) {
      SizeValues2.emplace_back(B.getArgument(Sizes2[CAxis]));
    }
    Value ToStore;
    auto ElementType = InType.getElementType().cast<FloatType>();
    Value ReductionVar =
        rewriter.create<arith::ConstantFloatOp>(Loc, APFloat(0.0), ElementType);
    SmallVector<Value> InductionVars2;
    for (auto SzValue : SizeValues2) {
      auto AffineFor = rewriter.create<AffineForOp>(
          Loc, ZeroIndexVar, rewriter.getSymbolIdentityMap(), SzValue,
          rewriter.getSymbolIdentityMap(), 1, ReductionVar);
      if (!ToStore) {
        ToStore = AffineFor.getResult(0);
        rewriter.create<memref::StoreOp>(Loc, ToStore, Alloc, InductionVars);
      } else {
        rewriter.create<AffineYieldOp>(Loc, AffineFor.getResults());
      }
      auto InnerMostBody = AffineFor.getBody();
      InductionVars2.emplace_back(AffineFor.getInductionVar());
      ReductionVar = AffineFor.getRegionIterArgs()[0];
      rewriter.setInsertionPointToStart(InnerMostBody);
    }
    Value LoadFrom = adaptor.getInput();
    SmallVector<Value> LoadIndexes;
    int64_t Idx1 = 0, Idx2 = 0;
    for (int64_t Axis = 0; Axis < (int64_t)(Sizes2.size()); Axis++) {
      if (llvm::none_of(ContractedAxes,
                        [&](auto Axis2) { return Axis == Axis2; })) {
        LoadIndexes.emplace_back(InductionVars[Idx1++]);
      } else {
        LoadIndexes.emplace_back(InductionVars2[Idx2++]);
      }
    }
    auto ToAdd = rewriter.create<memref::LoadOp>(Loc, LoadFrom, LoadIndexes);
    Value ToYield = rewriter.create<arith::AddFOp>(Loc, ReductionVar, ToAdd);
    rewriter.create<AffineYieldOp>(Loc, ToYield);
    rewriter.replaceOp(op, Alloc);
    return success();
  }
};

struct ContractionLoweringPass
    : public PassWrapper<ContractionLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ContractionLoweringPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, func::FuncDialect, memref::MemRefDialect>();
  }
  void runOnOperation() final;
};

static Value materializeToTensor(OpBuilder &builder, TensorType type,
                                 ValueRange inputs, Location loc) {
  assert(inputs.size() == 1);
  assert(inputs[0].getType().isa<BaseMemRefType>());
  auto UCCOp = builder.create<UnrealizedConversionCastOp>(
      loc, SmallVector<Type>(1, type), inputs);
  return UCCOp.getResult(0);
}

void ContractionLoweringPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addLegalDialect<AffineDialect, BuiltinDialect, arith::ArithDialect,
                         memref::MemRefDialect>();
  target.addIllegalDialect<TCDialect>();
  // target.addLegalOp<func::ReturnOp>();
  TypeConverter TC;
  TC.addConversion([](Type type) { return type; });
  TC.addConversion([](RankedTensorType type) -> Type {
    return MemRefType::get(type.getShape(), type.getElementType());
  });
  TC.addArgumentMaterialization(materializeToTensor);
  TC.addSourceMaterialization(materializeToTensor);
  TC.addTargetMaterialization([](OpBuilder &builder, BaseMemRefType type,
                                 ValueRange inputs, Location loc) -> Value {
    assert(inputs.size() == 1 && "expected exactly one input");
    if (inputs[0].getType().isa<TensorType>()) {
      // Tensor to MemRef cast.
      auto UCCOp = builder.create<UnrealizedConversionCastOp>(
          loc, SmallVector<Type>(1, type), inputs);
      return UCCOp.getResult(0);
    }
    llvm_unreachable("only tensor/memref input types supported");
  });
  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    return llvm::all_of(op.getFunctionType().getInputs(),
                        [&](Type T) { return TC.isLegal(T); });
  });
  target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
    return llvm::all_of(op.getOperandTypes(),
                        [&](Type T) { return TC.isLegal(T); });
  });
  RewritePatternSet patterns(&getContext());
  patterns.add<FuncOpLowering2, ReturnOpLowering, AddOpLowering, MulOpLowering,
               PermuteOpLowering, ContractOpLowering, DiagonalOpLowering,
               ConstantOpLowering, SliceOpLowering, NegOpLowering>(
      TC, &getContext());
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> createLowerToAffinePass() {
  return std::make_unique<ContractionLoweringPass>();
}
} // namespace tc
} // namespace mlir
