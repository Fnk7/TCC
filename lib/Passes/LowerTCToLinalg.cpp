#include "llvm/ADT/Sequence.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "TCC/Dialect/TC/IR/TCOps.h"
#include "TCC/Dialect/TC/Utils/Utils.h"
#include "TCC/Passes.h"

namespace mlir {
namespace tc {
// static SmallVector<int64_t> unpackIntegerArrayAttr(ArrayAttr Attr) {
//   SmallVector<int64_t> Sizes;
//   for (auto &I64 : Attr) {
//     Sizes.emplace_back(I64.cast<IntegerAttr>().getValue().getSExtValue());
//   }
//   return Sizes;
// }
// static SmallVector<int64_t> getTensorSizeIDs(RankedTensorType RTT) {
//   auto Attr = RTT.getEncoding().cast<ArrayAttr>();
//   return unpackIntegerArrayAttr(Attr);
// }
// static SmallVector<int64_t> getTensorSizes(Value V) {
//   auto RTT = V.getType().cast<RankedTensorType>();
//   return getTensorSizeIDs(RTT);
// }

// static std::tuple<Value, SmallVector<Value>>
// createBufferForTensor(OpBuilder &builder, Value tensorValue) {
//   auto loc = tensorValue.getLoc();
//   auto tensorType = cast<RankedTensorType>(tensorValue.getType());
//   auto memrefType =
//       MemRefType::get(tensorType.getShape(), tensorType.getElementType());
//   auto shape = tensorType.getShape();
//   auto sizeIDs = getTensorSizeIDs(tensorType);
//   assert(shape.size() == sizeIDs.size());
//   auto block = tensorValue.getParentBlock();
//   SmallVector<Value> allocSizes;
//   SmallVector<Value> sizeValues;
//   for (auto [size, sizeID] : llvm::zip(shape, sizeIDs)) {
//     if (size > 0) {
//       auto sizeValue = builder.create<arith::ConstantIndexOp>(loc, size);
//       sizeValues.emplace_back(sizeValue);
//     } else {
//       sizeValues.emplace_back(block->getArgument(sizeID));
//       allocSizes.emplace_back(sizeValues.back());
//     }
//   }
//   bool isAlloca = true;
//   for (auto U : tensorValue.getUsers()) {
//     if (dyn_cast<func::ReturnOp>(U)) {
//       isAlloca = false;
//     }
//   }
//   if (isAlloca) {
//     auto alloca = builder.create<memref::AllocaOp>(loc, memrefType,
//     allocSizes); return {alloca, sizeValues};
//   } else {
//     auto alloc = builder.create<memref::AllocOp>(loc, memrefType,
//     allocSizes); return {alloc, sizeValues};
//   }
// }

// static auto createNestedAffineForOps(OpBuilder &builder, Location loc,
//                                      ArrayRef<Value> sizeValues) {
//   Value zeroIndexValue = builder.create<arith::ConstantIndexOp>(loc, 0);
//   SmallVector<Value> inductionValues;
//   for (auto sizeValue : sizeValues) {
//     auto forOp = builder.create<AffineForOp>(
//         loc, zeroIndexValue, builder.getSymbolIdentityMap(), sizeValue,
//         builder.getSymbolIdentityMap(), 1);
//     inductionValues.emplace_back(forOp.getInductionVar());
//     builder.setInsertionPointToStart(forOp.getBody());
//   }
//   return inductionValues;
// }

static Value createEmptyTensorForLinalgOuts(OpBuilder &builder, Value value) {
  auto [shape, sizeIDs] = getTensorSizeInfos(value);
  auto RTT = value.getType().cast<RankedTensorType>();
  auto newType = RankedTensorType::get(RTT.getShape(), RTT.getElementType());
  auto *block = value.getParentBlock();
  SmallVector<Value> dynamicSizes;
  for (auto [size, sizeID] : llvm::zip(shape, sizeIDs)) {
    if (size < 0) {
      dynamicSizes.emplace_back(block->getArgument(sizeID));
    }
  }
  auto emptyOp =
      builder.create<tensor::EmptyOp>(value.getLoc(), newType, dynamicSizes);
  return emptyOp.getResult();
}

struct AddOpLowering : public OpConversionPattern<AddOp> {
  using OpConversionPattern<AddOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AddOp op, typename AddOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto out = createEmptyTensorForLinalgOuts(rewriter, op.getResult());
    auto ins = adaptor.getInputs();
    auto rank = getTensorRank(out);
    SmallVector<Value> outs(1, out);
    SmallVector<AffineMap> indexingMaps;
    indexingMaps.push_back(rewriter.getMultiDimIdentityMap(rank));
    indexingMaps.push_back(rewriter.getMultiDimIdentityMap(rank));
    indexingMaps.push_back(rewriter.getMultiDimIdentityMap(rank));
    SmallVector<StringRef> iteratorTypes(rank, StringRef("parallel"));
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, out.getType(), ins, outs, indexingMaps, iteratorTypes,
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          auto numArgs = args.size();
          Value toYield = args[0];
          for (auto idx : llvm::seq<int64_t>(1, numArgs - 1)) {
            auto addOp = builder.create<arith::AddFOp>(loc, toYield, args[idx]);
            toYield = addOp.getResult();
          }
          builder.create<linalg::YieldOp>(loc, toYield);
        });
    rewriter.replaceOp(op, genericOp.getResults());
    return success();
  }
};

// using AddOpLowering = BinaryOpLowering<AddOp, arith::AddFOp>;
// using MulOpLowering = BinaryOpLowering<MulOp, arith::MulFOp>;

struct ReturnOpLowering : public OpConversionPattern<func::ReturnOp> {
  using OpConversionPattern<func::ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp op, func::ReturnOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.updateRootInPlace(op,
                               [&] { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};

// struct ConstantOpLowering : public OpConversionPattern<ConstantOp> {
//   using OpConversionPattern<ConstantOp>::OpConversionPattern;

//   LogicalResult
//   matchAndRewrite(ConstantOp op, ConstantOp::Adaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const final {
//     auto loc = op.getLoc();
//     auto memrefType =
//         MemRefType::get(op.getType().getShape(),
//         op.getType().getElementType());
//     auto allocaOp = rewriter.create<memref::AllocaOp>(loc, memrefType,
//                                                       SmallVector<Value>());
//     auto constOp = rewriter.create<arith::ConstantFloatOp>(
//         loc, adaptor.getValue(), rewriter.getF64Type());
//     rewriter.create<memref::StoreOp>(loc, constOp, allocaOp);
//     rewriter.replaceOp(op, allocaOp.getResult());
//     return success();
//   }
// };

// struct ReturnOpLowering2 : public OpConversionPattern<func::ReturnOp> {
//   using OpConversionPattern<func::ReturnOp>::OpConversionPattern;

//   LogicalResult
//   matchAndRewrite(func::ReturnOp op, func::ReturnOp::Adaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const final {
//     rewriter.updateRootInPlace(op,
//                                [&] { op->setOperands(adaptor.getOperands());
//                                });
//     return success();
//   }
// };

struct FuncOpLowering : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(func::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = rewriter.getUnknownLoc();
    auto funcType = op.getFunctionType();
    TypeConverter::SignatureConversion signature(funcType.getNumInputs());
    for (auto [argIdx, argType] : llvm::enumerate(funcType.getInputs())) {
      if (const auto &RTT = argType.dyn_cast_or_null<RankedTensorType>()) {
        auto newType = typeConverter->convertType(RTT);
        signature.addInputs(argIdx, newType);
      } else {
        signature.addInputs(argIdx, argType);
      }
    }
    SmallVector<Type> resultTypes;
    for (auto resultType : funcType.getResults()) {
      const auto &RTT = resultType.cast<RankedTensorType>();
      auto newType =
          RankedTensorType::get(RTT.getShape(), RTT.getElementType());
      resultTypes.push_back(newType);
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
    if (failed(rewriter.convertRegionTypes(&op.getRegion(), *typeConverter)))
      return failure();
    rewriter.inlineRegionBefore(op.getRegion(), newFuncOp.getRegion(),
                                newFuncOp.getRegion().end());
    rewriter.eraseOp(op);
    return success();
  }
};

// struct PermuteOpLowering : public OpConversionPattern<PermuteOp> {
//   using OpConversionPattern<PermuteOp>::OpConversionPattern;

//   LogicalResult
//   matchAndRewrite(PermuteOp op, PermuteOp::Adaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const final {
//     auto Loc = op.getLoc();
//     auto [buffer, SizeValues] = createBufferForTensor(rewriter,
//     op.getResult()); auto InductionVars = createNestedAffineForOps(rewriter,
//     Loc, SizeValues); Value LoadFrom = adaptor.getInput(); SmallVector<Value>
//     Indexes; auto AxesMapAttr = op.getAxesMap(); auto AxesMap =
//     unpackIntegerArrayAttr(AxesMapAttr); for (auto Axis : AxesMap) {
//       Indexes.emplace_back(InductionVars[Axis]);
//     }
//     Value ToStore = rewriter.create<memref::LoadOp>(Loc, LoadFrom, Indexes);
//     rewriter.create<memref::StoreOp>(Loc, ToStore, buffer, InductionVars);
//     rewriter.replaceOp(op, buffer);
//     return success();
//   }
// };

// struct DiagonalOpLowering : public OpConversionPattern<DiagonalOp> {
//   using OpConversionPattern<DiagonalOp>::OpConversionPattern;

//   LogicalResult
//   matchAndRewrite(DiagonalOp op, DiagonalOp::Adaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const final {
//     auto Loc = op.getLoc();
//     auto [Alloc, SizeValues] = createBufferForTensor(rewriter,
//     op.getResult()); SmallVector<Value> InductionVars =
//         createNestedAffineForOps(rewriter, Loc, SizeValues);
//     Value LoadFrom = adaptor.getInput();
//     SmallVector<Value> Indexes(
//         op.getOperand().getType().cast<RankedTensorType>().getShape().size());
//     auto AxesMapAttr = op.getAxesPairs();
//     for (auto [IndexValue, AxesAttr] : llvm::zip(InductionVars, AxesMapAttr))
//     {
//       auto Axes = unpackIntegerArrayAttr(AxesAttr.cast<ArrayAttr>());
//       for (auto Axis : Axes) {
//         Indexes[Axis] = IndexValue;
//       }
//     }
//     Value ToStore = rewriter.create<memref::LoadOp>(Loc, LoadFrom, Indexes);
//     rewriter.create<memref::StoreOp>(Loc, ToStore, Alloc, InductionVars);
//     rewriter.replaceOp(op, Alloc);
//     return success();
//   }
// };

// struct SliceOpLowering : public OpConversionPattern<SliceOp> {
//   using OpConversionPattern<SliceOp>::OpConversionPattern;

//   LogicalResult
//   matchAndRewrite(SliceOp op, SliceOp::Adaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const final {
//     auto Loc = op.getLoc();
//     auto [Alloc, SizeValues] = createBufferForTensor(rewriter,
//     op.getResult()); SmallVector<Value> InductionVars =
//         createNestedAffineForOps(rewriter, Loc, SizeValues);
//     Value LoadFrom = adaptor.getInput();
//     SmallVector<Value> LoadIndices(
//         op.getOperand(0).getType().cast<RankedTensorType>().getShape().size());
//     auto AxesAttr = op.getAxesAttr();
//     for (auto [Axis, IndexValue] :
//          llvm::zip(unpackIntegerArrayAttr(AxesAttr), op.getIndices())) {
//       LoadIndices[Axis] = IndexValue;
//     }
//     unsigned IndexPos = 0;
//     for (auto IndexValue : InductionVars) {
//       while (LoadIndices[IndexPos]) {
//         IndexPos++;
//       }
//       LoadIndices[IndexPos] = IndexValue;
//     }
//     Value ToStore = rewriter.create<memref::LoadOp>(Loc, LoadFrom,
//     LoadIndices); rewriter.create<memref::StoreOp>(Loc, ToStore, Alloc,
//     InductionVars); rewriter.replaceOp(op, Alloc); return success();
//   }
// };

// struct NegOpLowering : public OpConversionPattern<NegOp> {
//   using OpConversionPattern<NegOp>::OpConversionPattern;

//   LogicalResult
//   matchAndRewrite(NegOp op, NegOp::Adaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const final {
//     auto Loc = op.getLoc();
//     auto [alloc, sizeValues] = createBufferForTensor(rewriter,
//     op.getResult()); auto InductionVars = createNestedAffineForOps(rewriter,
//     Loc, sizeValues); Value LoadFrom = adaptor.getInput(); Value ToStore =
//         rewriter.create<memref::LoadOp>(Loc, LoadFrom, InductionVars);
//     ToStore = rewriter.create<arith::NegFOp>(Loc, ToStore);
//     rewriter.create<memref::StoreOp>(Loc, ToStore, alloc, InductionVars);
//     rewriter.replaceOp(op, alloc);
//     return success();
//   }
// };

// struct ContractOpLowering : public OpConversionPattern<ContractOp> {
//   using OpConversionPattern<ContractOp>::OpConversionPattern;

//   LogicalResult
//   matchAndRewrite(ContractOp op, ContractOp::Adaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const final {
//     auto Loc = op.getLoc();
//     auto [Alloc, SizeValues] = createBufferForTensor(rewriter,
//     op.getResult()); auto InType =
//     op.getResult().getType().cast<RankedTensorType>(); auto &B =
//     *(op->getBlock()); Value ZeroIndexVar =
//     rewriter.create<arith::ConstantIndexOp>(Loc, 0); SmallVector<Value>
//     InductionVars =
//         createNestedAffineForOps(rewriter, Loc, SizeValues);
//     // Inner Reduction Loop
//     auto Sizes2 = getTensorSizes(op.getOperand());
//     auto AxesAttr = op.getAxesAttr();
//     auto ContractedAxes = unpackIntegerArrayAttr(AxesAttr);
//     SmallVector<Value> SizeValues2;
//     for (auto CAxis : ContractedAxes) {
//       SizeValues2.emplace_back(B.getArgument(Sizes2[CAxis]));
//     }
//     Value ToStore;
//     auto ElementType = InType.getElementType().cast<FloatType>();
//     Value ReductionVar =
//         rewriter.create<arith::ConstantFloatOp>(Loc, APFloat(0.0),
//         ElementType);
//     SmallVector<Value> InductionVars2;
//     for (auto SzValue : SizeValues2) {
//       auto AffineFor = rewriter.create<AffineForOp>(
//           Loc, ZeroIndexVar, rewriter.getSymbolIdentityMap(), SzValue,
//           rewriter.getSymbolIdentityMap(), 1, ReductionVar);
//       if (!ToStore) {
//         ToStore = AffineFor.getResult(0);
//         rewriter.create<memref::StoreOp>(Loc, ToStore, Alloc, InductionVars);
//       } else {
//         rewriter.create<AffineYieldOp>(Loc, AffineFor.getResults());
//       }
//       auto InnerMostBody = AffineFor.getBody();
//       InductionVars2.emplace_back(AffineFor.getInductionVar());
//       ReductionVar = AffineFor.getRegionIterArgs()[0];
//       rewriter.setInsertionPointToStart(InnerMostBody);
//     }
//     Value LoadFrom = adaptor.getInput();
//     SmallVector<Value> LoadIndexes;
//     int64_t Idx1 = 0, Idx2 = 0;
//     for (int64_t Axis = 0; Axis < (int64_t)(Sizes2.size()); Axis++) {
//       if (llvm::none_of(ContractedAxes,
//                         [&](auto Axis2) { return Axis == Axis2; })) {
//         LoadIndexes.emplace_back(InductionVars[Idx1++]);
//       } else {
//         LoadIndexes.emplace_back(InductionVars2[Idx2++]);
//       }
//     }
//     auto ToAdd = rewriter.create<memref::LoadOp>(Loc, LoadFrom, LoadIndexes);
//     Value ToYield = rewriter.create<arith::AddFOp>(Loc, ReductionVar, ToAdd);
//     rewriter.create<AffineYieldOp>(Loc, ToYield);
//     rewriter.replaceOp(op, Alloc);
//     return success();
//   }
// };

struct ConvertTCToLinalgPass
    : public PassWrapper<ConvertTCToLinalgPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertTCToLinalgPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, memref::MemRefDialect,
                    linalg::LinalgDialect>();
  }
  void runOnOperation() final;
};

static Value materializeToTensor(OpBuilder &builder, TensorType type,
                                 ValueRange inputs, Location loc) {
  assert(inputs.size() == 1);
  assert(inputs[0].getType().isa<RankedTensorType>());
  auto castOp = builder.create<UnrealizedConversionCastOp>(
      loc, SmallVector<Type>(1, type), inputs);
  return castOp.getResult(0);
}

void ConvertTCToLinalgPass::runOnOperation() {
  auto op = getOperation();
  ConversionTarget target(getContext());
  target.addLegalDialect<BuiltinDialect, arith::ArithDialect,
                         memref::MemRefDialect, linalg::LinalgDialect,
                         tensor::TensorDialect, func::FuncDialect>();
  target.addIllegalDialect<TCDialect>();
  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    return llvm::all_of(op.getFunctionType().getInputs(), [&](Type T) {
      if (auto RTT = T.dyn_cast_or_null<RankedTensorType>()) {
        if (RTT.getEncoding()) {
          return false;
        }
      }
      return true;
    });
  });
  target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
    return llvm::all_of(op.getOperands(), [&](Value value) {
      if (auto RTT = value.getType().dyn_cast_or_null<RankedTensorType>()) {
        if (RTT.getEncoding()) {
          return false;
        }
      }
      return true;
    });
  });
  TypeConverter typeConverter;
  typeConverter.addConversion([](Type type) { return type; });
  typeConverter.addConversion([](RankedTensorType type) -> Type {
    return RankedTensorType::get(type.getShape(), type.getElementType());
  });
  typeConverter.addArgumentMaterialization(materializeToTensor);
  typeConverter.addSourceMaterialization(materializeToTensor);
  typeConverter.addTargetMaterialization(
      [](OpBuilder &builder, RankedTensorType type, ValueRange inputs,
         Location loc) -> Value {
        assert(inputs.size() == 1 && "expected exactly one input");
        if (inputs[0].getType().isa<TensorType>()) {
          auto castOp = builder.create<UnrealizedConversionCastOp>(
              loc, SmallVector<Type>(1, type), inputs);
          return castOp.getResult(0);
        }
        llvm_unreachable("only tensor/memref input types supported");
      });
  RewritePatternSet patterns(&getContext());
  patterns.add<FuncOpLowering>(typeConverter, patterns.getContext());
  patterns.add<ReturnOpLowering>(typeConverter, patterns.getContext());
  patterns.add<AddOpLowering>(typeConverter, patterns.getContext());
  // patterns.add<FuncOpLowering2, ReturnOpLowering, MulOpLowering,
  //              PermuteOpLowering, ContractOpLowering, DiagonalOpLowering,
  //              ConstantOpLowering, SliceOpLowering, NegOpLowering>();
  if (failed(applyPartialConversion(op, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> createConvertTCToLinalgPass() {
  return std::make_unique<ConvertTCToLinalgPass>();
}
} // namespace tc
} // namespace mlir
