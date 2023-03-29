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
#include "TCC/Dialect/Utils/FuncUtils.h"
#include "TCC/Passes.h"

// #define TBLAS_EMIT_C

namespace mlir {
namespace tc {
namespace {
auto getBatchTensorType(OpBuilder &builder, Type type) {
  auto RTT = type.cast<RankedTensorType>();
  auto shape = RTT.getShape();
  if (shape.size() == 0)
    return type;
  auto sizeIDs = getTensorSizeIDs(RTT);
  auto symmetry = getTensorSymmetry(RTT);
  TShapeTy newShape(1, -1);
  newShape.append(shape.begin(), shape.end());
  SizeIDsTy newSizeIDs(1, 0);
  for (auto sizeID : sizeIDs)
    newSizeIDs.emplace_back(sizeID + 1);
  SymmArrayTy newSymmetry;
  for (auto pair : symmetry)
    newSymmetry.emplace_back(pair.first + 1, pair.second + 1);
  return getEncodedTensorType(builder, newShape, newSizeIDs, newSymmetry);
}

AxesMapTy getNewAxesMap(Value input, AxesMapRefTy axesMap) {
  auto rank = getTensorRank(input);
  if (axesMap.size() != (size_t)rank) {
    AxesMapTy newAxesMap(1, 0);
    for (auto axis : axesMap)
      newAxesMap.emplace_back(axis + 1);
    return newAxesMap;
  }
  return AxesMapTy(axesMap);
}

template <typename TCOp>
struct AbelianOpLowering : public OpConversionPattern<TCOp> {
  using OpConversionPattern<TCOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(TCOp op, typename TCOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto inputs = adaptor.getInputs();
    auto axesMaps = op.getAxesMaps();
    AxesMapsTy newAxesMaps;
    for (auto [input, axesMap] : llvm::zip(inputs, axesMaps))
      newAxesMaps.emplace_back(getNewAxesMap(input, axesMap));
    rewriter.replaceOpWithNewOp<TCOp>(op, inputs, newAxesMaps);
    return success();
  }
};

using AddOpLowering = AbelianOpLowering<AddOp>;
using MulOpLowering = AbelianOpLowering<MulOp>;

struct ReturnOpLowering : public OpConversionPattern<ReturnOp> {
  using OpConversionPattern<ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReturnOp op, ReturnOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<ReturnOp>(op, adaptor.getOperands());
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

struct PermuteOpLowering : public OpConversionPattern<PermuteOp> {
  using OpConversionPattern<PermuteOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PermuteOp op, PermuteOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto input = adaptor.getInput();
    auto axesMap = op.getAxesMapArray();
    AxesMapTy newAxesMap = getNewAxesMap(input, axesMap);
    rewriter.replaceOpWithNewOp<PermuteOp>(op, adaptor.getInput(), newAxesMap);
    return success();
  }
};

struct ContractOpLowering : public OpConversionPattern<ContractOp> {
  using OpConversionPattern<ContractOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ContractOp op, ContractOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto axes = op.getAxes();
    auto newAxes = llvm::to_vector(
        llvm::map_range(axes, [](auto axis) { return axis + 1; }));
    rewriter.replaceOpWithNewOp<ContractOp>(op, adaptor.getInput(), newAxes);
    return success();
  }
};

void addBatchSizeAsArgument(func::FuncOp funcOp) {
  auto builder = OpBuilder(funcOp);
  auto funcType = funcOp.getFunctionType();
  SmallVector<Type> inputTypes(1, builder.getIndexType());
  inputTypes.append(funcType.getInputs().begin(), funcType.getInputs().end());
  auto newFuncType = builder.getFunctionType(inputTypes, funcType.getResults());
  funcOp.setType(newFuncType);
  funcOp.front().insertArgument((unsigned)0, inputTypes[0], funcOp.getLoc());
  for (auto argIdx = inputTypes.size() - 1; argIdx >= 1; argIdx--)
    funcOp.setArgAttrs(argIdx, funcOp.getArgAttrs(argIdx - 1));
  funcOp.setArgAttr(0, "func.name", builder.getStringAttr("batch#size"));
  auto numSizes = utils::getNumInputSizes(funcOp);
  auto numSizesAttr = builder.getI64IntegerAttr(numSizes + 1);
  utils::setNumInputSizes(funcOp, numSizesAttr);
}

struct AddBatchPass
    : public PassWrapper<AddBatchPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AddBatchPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<tensor::TensorDialect>();
  }
  void runOnOperation() final;
};

void AddBatchPass::runOnOperation() {
  auto op = getOperation();
  auto *ctx = op.getContext();
  auto builder = OpBuilder(op);
  ConversionTarget target(*ctx);
  target.addLegalDialect<BuiltinDialect>();
  target.addLegalDialect<arith::ArithDialect>();
  target.addIllegalDialect<TCDialect>();
  SetVector<Operation *> ops;
  for (auto funcOp : op.getOps<func::FuncOp>()) {
    ops.insert(funcOp);
    for (auto &op : funcOp.front()) {
      ops.insert(&op);
    }
  }
  target.addDynamicallyLegalDialect<TCDialect>([&](Operation *op) {
    if (ops.count(op) && !dyn_cast_or_null<ConstantOp>(op))
      return false;
    return true;
  });
  target.addDynamicallyLegalOp<func::FuncOp>(
      [&](func::FuncOp op) { return !ops.count(op); });
  // Type Conversion, Tensor Encoding
  TypeConverter typeConverter;
  typeConverter.addConversion([](Type type) { return type; });
  typeConverter.addConversion([&](RankedTensorType type) -> Type {
    return getBatchTensorType(builder, type);
  });
  // Materialization batched <-> not batched
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
  if (failed(applyPartialConversion(op, target, std::move(patterns))))
    signalPassFailure();
  for (auto funcOp : op.getOps<func::FuncOp>())
    addBatchSizeAsArgument(funcOp);
}
} // namespace
std::unique_ptr<Pass> createAddBatchPass() {
  return std::make_unique<AddBatchPass>();
}
} // namespace tc
} // namespace mlir
