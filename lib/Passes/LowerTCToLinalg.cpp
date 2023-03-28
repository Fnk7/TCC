#include "llvm/ADT/Sequence.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "TCC/Dialect/TC/IR/TCOps.h"
#include "TCC/Dialect/TC/Utils/Utils.h"
#include "TCC/Passes.h"

// #define TBLAS_EMIT_C

namespace mlir {
namespace tc {
auto getAffineMapsByAxesMaps(int64_t rank, const AxesMapsTy &axesMaps,
                             MLIRContext *ctx) {
  SmallVector<AffineMap> maps;
  for (auto axesMap : axesMaps) {
    SmallVector<AffineExpr> exprs;
    for (auto axis : axesMap) {
      exprs.emplace_back(getAffineDimExpr(axis, ctx));
    }
    maps.emplace_back(AffineMap::get(rank, 0, exprs, ctx));
  }
  return maps;
}

static Value createEmptyTensorForLinalgOuts(OpBuilder &builder,
                                            TypeConverter *typeConverter,
                                            Value value) {
  auto RTT = typeConverter->convertType(value.getType());
  auto *block = value.getParentBlock();
  SmallVector<Value> dynamicSizes;
  auto [shape, sizeIDs] = getTensorSizeInfos(value);
  for (auto [size, sizeID] : llvm::zip(shape, sizeIDs)) {
    if (size < 0)
      dynamicSizes.emplace_back(block->getArgument(sizeID));
  }
  auto emptyOp =
      builder.create<tensor::EmptyOp>(value.getLoc(), RTT, dynamicSizes);
  return emptyOp.getResult();
}

template <typename TCOp, typename ArithOp>
struct AbelianOpLowering : public OpConversionPattern<TCOp> {
  using OpConversionPattern<TCOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(TCOp op, typename TCOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto *ctx = op.getContext();
    auto loc = op.getLoc();
    auto axesMaps = getAxesMaps(op);
    auto out = createEmptyTensorForLinalgOuts(rewriter, this->typeConverter,
                                              op.getResult());
    auto ins = adaptor.getInputs();
    auto rank = getTensorRank(out);
    SmallVector<Value> outs(1, out);
    SmallVector<AffineMap> indexingMaps =
        getAffineMapsByAxesMaps(rank, axesMaps, ctx);
    // result tensor
    indexingMaps.push_back(rewriter.getMultiDimIdentityMap(rank));
    SmallVector<StringRef> iteratorTypes(rank, StringRef("parallel"));
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, out.getType(), ins, outs, indexingMaps, iteratorTypes,
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          auto numArgs = args.size();
          Value toYield = args[0];
          for (auto idx : llvm::seq<int64_t>(1, numArgs - 1)) {
            auto addOp = builder.create<ArithOp>(loc, toYield, args[idx]);
            toYield = addOp.getResult();
          }
          builder.create<linalg::YieldOp>(loc, toYield);
        });
    rewriter.replaceOp(op, genericOp.getResults());
    return success();
  }
};

using AddOpLowering = AbelianOpLowering<AddOp, arith::AddFOp>;
using MulOpLowering = AbelianOpLowering<MulOp, arith::MulFOp>;

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

struct FuncOpLowering : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(func::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = rewriter.getUnknownLoc();
    auto funcType = op.getFunctionType();
    TypeConverter::SignatureConversion signature(funcType.getNumInputs());
    if (failed(typeConverter->convertSignatureArgs(funcType.getInputs(),
                                                   signature)))
      return failure();
    if (failed(rewriter.convertRegionTypes(&op.getBody(), *typeConverter,
                                           &signature)))
      return failure();
    auto resultTypes =
        llvm::to_vector(llvm::map_range(funcType.getResults(), [&](Type type) {
          return typeConverter->convertType(type);
        }));
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
    rewriter.inlineRegionBefore(op.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());
    rewriter.eraseOp(op);
    return success();
  }
};

auto getAffineMapByAxesMap(AxesMapTy &axesMap, MLIRContext *ctx) {
  auto rank = axesMap.size();
  SmallVector<AffineExpr> exprs;
  for (auto axis : axesMap) {
    exprs.emplace_back(getAffineDimExpr(axis, ctx));
  }
  return AffineMap::get(rank, 0, exprs, ctx);
}

struct PermuteOpLowering : public OpConversionPattern<PermuteOp> {
  using OpConversionPattern<PermuteOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PermuteOp op, PermuteOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto *ctx = op.getContext();
    auto loc = op.getLoc();
    auto input = adaptor.getInput();
    auto output =
        createEmptyTensorForLinalgOuts(rewriter, typeConverter, op.getResult());
    auto axesMap = op.getAxesMapArray();
    auto rank = axesMap.size();
    SmallVector<Value> ins(1, input);
    SmallVector<Value> outs(1, output);
    SmallVector<AffineMap> indexingMaps;
    indexingMaps.push_back(getAffineMapByAxesMap(axesMap, ctx));
    indexingMaps.push_back(rewriter.getMultiDimIdentityMap(rank));
    SmallVector<StringRef> iteratorTypes(rank, StringRef("parallel"));
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, output.getType(), ins, outs, indexingMaps, iteratorTypes,
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          builder.create<linalg::YieldOp>(loc, args[0]);
        });
    rewriter.replaceOp(op, genericOp.getResults());
    return success();
  }
};

auto getAffineMapByAxes(int64_t rank, AxesTy &axes, MLIRContext *ctx) {
  SmallVector<AffineExpr> exprs;
  auto i2oAxesMap = getContractOpInput2OutputAxisMap(axes, rank);
  for (auto [iAxis, oAxis] : llvm::enumerate(i2oAxesMap)) {
    if (oAxis >= 0) {
      exprs.emplace_back(getAffineDimExpr(iAxis, ctx));
    }
  }
  return AffineMap::get(rank, 0, exprs, ctx);
}

static Value createLinalgFillOp(OpBuilder &builder,
                                TypeConverter *typeConverter, Value value,
                                APFloat cst = APFloat(0.0)) {
  auto zero = builder.create<arith::ConstantFloatOp>(value.getLoc(), cst,
                                                     builder.getF64Type());
  auto out = createEmptyTensorForLinalgOuts(builder, typeConverter, value);
  SmallVector<Value> inputs(1, zero.getResult());
  SmallVector<Value> outputs(1, out);
  auto fillOp = builder.create<linalg::FillOp>(value.getLoc(), inputs, outputs);
  return fillOp.getResult(0);
}

struct ContractOpLowering : public OpConversionPattern<ContractOp> {
  using OpConversionPattern<ContractOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ContractOp op, ContractOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto *ctx = op.getContext();
    auto loc = op.getLoc();
    auto input = adaptor.getInput();
    auto rank = getTensorRank(input);
    auto axes = op.getAxesArray();
    auto out = createLinalgFillOp(rewriter, typeConverter, op.getResult());
    SmallVector<Value> ins(1, input);
    SmallVector<Value> outs(1, out);
    // input tensor
    SmallVector<AffineMap> indexingMaps;
    indexingMaps.push_back(rewriter.getMultiDimIdentityMap(rank));
    indexingMaps.push_back(getAffineMapByAxes(rank, axes, ctx));
    SmallVector<StringRef> iteratorTypes(rank, StringRef("parallel"));
    for (auto axis : axes)
      iteratorTypes[axis] = "reduction";
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, out.getType(), ins, outs, indexingMaps, iteratorTypes,
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          auto toYield = builder.create<arith::AddFOp>(loc, args[0], args[1]);
          builder.create<linalg::YieldOp>(loc, toYield.getResult());
        });
    rewriter.replaceOp(op, genericOp.getResults());
    return success();
  }
};

struct ConstantOpLowering : public OpConversionPattern<ConstantOp> {
  using OpConversionPattern<ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp op, ConstantOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto value = createLinalgFillOp(rewriter, typeConverter, op.getResult(),
                                    adaptor.getValue());
    rewriter.replaceOp(op, value);
    return success();
  }
};

struct ConvertTCToLinalgPass
    : public PassWrapper<ConvertTCToLinalgPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertTCToLinalgPass)

  bool eraseTensorEncoding;
  ConvertTCToLinalgPass(bool eraseTensorEncoding)
      : eraseTensorEncoding(eraseTensorEncoding) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<tensor::TensorDialect>();
  }
  void runOnOperation() final;
};

void ConvertTCToLinalgPass::runOnOperation() {
  auto op = getOperation();
  auto *ctx = op.getContext();
  ConversionTarget target(*ctx);
  target.addLegalDialect<BuiltinDialect>();
  target.addLegalDialect<func::FuncDialect>();
  target.addLegalDialect<linalg::LinalgDialect>();
  target.addLegalDialect<arith::ArithDialect>();
  target.addLegalDialect<tensor::TensorDialect>();
  target.addIllegalDialect<TCDialect>();
  if (eraseTensorEncoding) {
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return llvm::all_of(op.getFunctionType().getInputs(), [&](Type type) {
        auto RTT = type.dyn_cast_or_null<RankedTensorType>();
        return !(RTT && RTT.getEncoding());
      });
    });
    target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
      return llvm::all_of(op.getOperands(), [&](Value value) {
        auto RTT = value.getType().dyn_cast_or_null<RankedTensorType>();
        return !(RTT && RTT.getEncoding());
      });
    });
  }
  // Type Conversion, Tensor Encoding
  TypeConverter typeConverter;
  typeConverter.addConversion([](Type type) { return type; });
  if (eraseTensorEncoding) {
    typeConverter.addConversion([](RankedTensorType type) -> Type {
      return RankedTensorType::get(type.getShape(), type.getElementType());
    });
  }
  // Materialization encoding <-> not encoding
  auto materialize = [](OpBuilder &builder, RankedTensorType type,
                        ValueRange inputs, Location loc) -> Value {
    assert(inputs.size() == 1 && "Expected exactly one input");
    assert(inputs[0].getType().isa<RankedTensorType>() &&
           "Expected input type is RankedTensorType");
    auto castOp = builder.create<UnrealizedConversionCastOp>(
        loc, SmallVector<Type>(1, type), inputs);
    return castOp.getResult(0);
  };
  typeConverter.addArgumentMaterialization(materialize);
  typeConverter.addSourceMaterialization(materialize);
  typeConverter.addTargetMaterialization(materialize);
  // Patterns
  RewritePatternSet patterns(&getContext());
  patterns.add<FuncOpLowering>(typeConverter, ctx);
  patterns.add<ReturnOpLowering>(typeConverter, ctx);
  patterns.add<AddOpLowering>(typeConverter, ctx);
  patterns.add<MulOpLowering>(typeConverter, ctx);
  patterns.add<ContractOpLowering>(typeConverter, ctx);
  patterns.add<PermuteOpLowering>(typeConverter, ctx);
  patterns.add<ConstantOpLowering>(typeConverter, ctx);
  if (failed(applyPartialConversion(op, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> createConvertTCToLinalgPass(bool eraseTensorEncoding) {
  return std::make_unique<ConvertTCToLinalgPass>(eraseTensorEncoding);
}
} // namespace tc
} // namespace mlir
