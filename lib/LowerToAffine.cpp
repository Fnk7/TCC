#include "llvm/ADT/Sequence.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Passes.h"
#include "TCDialect/ContractionOps.h"

using namespace mlir;

namespace {
template <typename BinaryOp, typename LoweredBinaryOp>
struct BinaryOpLowering : public OpConversionPattern<BinaryOp> {
  using OpConversionPattern<BinaryOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BinaryOp op, typename BinaryOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<LoweredBinaryOp>(op, adaptor.getLhs(),
                                                 adaptor.getRhs());
    return success();
  }
};

using AddOpLowering = BinaryOpLowering<contraction::AddOp, arith::AddFOp>;
using MulOpLowering = BinaryOpLowering<contraction::MulOp, arith::MulFOp>;

struct ForOpLowering : public OpConversionPattern<contraction::ForOp> {
  using OpConversionPattern<contraction::ForOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(contraction::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto Loc = op.getLoc();
    auto Ranges = adaptor.getRanges();
    auto InType = op.getType().cast<TensorType>();
    auto OutType = MemRefType::get(InType.getShape(), InType.getElementType());
    auto Alloc = rewriter.create<memref::AllocOp>(Loc, OutType, Ranges);
    auto *ParentB = Alloc->getBlock();
    Alloc->moveBefore(&ParentB->front());
    bool NeedDealloc = true;
    for (auto *U : op.getResult().getUsers()) {
      if (dyn_cast<func::ReturnOp>(U))
        NeedDealloc = false;
    }
    if (NeedDealloc) {
      auto Dealloc = rewriter.create<memref::DeallocOp>(Loc, Alloc);
      Dealloc->moveBefore(&ParentB->back());
    }
    Value ZeroIndexVar = rewriter.create<arith::ConstantIndexOp>(Loc, 0);
    SmallVector<Value> NewRanges;
    Block *InnerMostBody;
    for (auto Range : Ranges) {
      auto AffineFor = rewriter.create<AffineForOp>(
          Loc, ZeroIndexVar, rewriter.getSymbolIdentityMap(), Range,
          rewriter.getSymbolIdentityMap(), 1);
      InnerMostBody = AffineFor.getBody();
      NewRanges.emplace_back(AffineFor.getInductionVar());
      rewriter.setInsertionPointToStart(InnerMostBody);
    }
    auto OldTerm = InnerMostBody->getTerminator();
    rewriter.eraseOp(OldTerm);
    rewriter.mergeBlocks(op.getBody(), InnerMostBody, NewRanges);
    auto Term = InnerMostBody->getTerminator();
    rewriter.setInsertionPoint(Term);
    rewriter.create<memref::StoreOp>(Loc, Term->getOperands()[0], Alloc,
                                     NewRanges);
    rewriter.replaceOpWithNewOp<AffineYieldOp>(Term);
    rewriter.replaceOp(op, Alloc->getResult(0));
    return success();
  }
};

struct ContractOpLowering
    : public OpConversionPattern<contraction::ContractOp> {
  using OpConversionPattern<contraction::ContractOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(contraction::ContractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto Loc = op.getLoc();
    auto Ranges = adaptor.getRanges();
    auto ElementType = op.getType();
    Value Range = Ranges[0];
    Value ZeroIndexVar = rewriter.create<arith::ConstantIndexOp>(Loc, 0);
    Value ReductionVar =
        rewriter.create<arith::ConstantFloatOp>(Loc, APFloat(0.0), ElementType);
    auto OuterMostFor = rewriter.create<AffineForOp>(
        Loc, ZeroIndexVar, rewriter.getSymbolIdentityMap(), Range,
        rewriter.getSymbolIdentityMap(), 1, ReductionVar);
    auto InnerMostFor = OuterMostFor;
    auto InnerMostBody = InnerMostFor.getBody();
    SmallVector<Value> NewRanges;
    NewRanges.emplace_back(InnerMostFor.getInductionVar());
    ReductionVar = InnerMostFor.getRegionIterArgs()[0];
    for (size_t i = 1; i < Ranges.size(); i++) {
      Range = Ranges[i];
      rewriter.setInsertionPointToStart(InnerMostBody);
      InnerMostFor = rewriter.create<AffineForOp>(
          Loc, ZeroIndexVar, rewriter.getSymbolIdentityMap(), Range,
          rewriter.getSymbolIdentityMap(), 1, ReductionVar);
      rewriter.create<AffineYieldOp>(Loc, InnerMostFor.getResults());
      InnerMostBody = InnerMostFor.getBody();
      NewRanges.emplace_back(InnerMostFor.getInductionVar());
      ReductionVar = InnerMostFor.getRegionIterArgs()[0];
    }
    rewriter.mergeBlocks(op.getBody(), InnerMostBody, NewRanges);
    auto Term = InnerMostBody->getTerminator();
    rewriter.setInsertionPoint(Term);
    Value YieldValue = rewriter.create<arith::AddFOp>(
        Loc, Term->getOperands()[0], ReductionVar);
    rewriter.replaceOpWithNewOp<AffineYieldOp>(Term, YieldValue);
    rewriter.replaceOp(op, OuterMostFor.getResults());
    return success();
  }
};

struct IndexOpLowering : public OpConversionPattern<contraction::IndexOp> {
  using OpConversionPattern<contraction::IndexOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(contraction::IndexOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<memref::LoadOp>(op, adaptor.getTensor(),
                                                adaptor.getIndexes());
    return success();
  }
};

struct ReturnOpLowering : public OpConversionPattern<func::ReturnOp> {
  using OpConversionPattern<func::ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
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
    auto InputTypes = op.getArgumentTypes();
    auto ResultTypes = op.getResultTypes();
    SmallVector<Type> NewInputTypes;
    SmallVector<Type> NewResultTypes;
    for (auto T : InputTypes) {
      if (auto TT = T.dyn_cast<TensorType>()) {
        NewInputTypes.emplace_back(
            MemRefType::get(TT.getShape(), TT.getElementType()));
      } else {
        NewInputTypes.emplace_back(T);
      }
    }
    for (auto T : ResultTypes) {
      auto TT = T.cast<TensorType>();
      NewResultTypes.emplace_back(
          MemRefType::get(TT.getShape(), TT.getElementType()));
    }
    auto NewFuncType = rewriter.getFunctionType(NewInputTypes, NewResultTypes);
    rewriter.startRootUpdate(op);
    op.setType(NewFuncType);
    auto &R = op.getRegion();
    unsigned NumArgs = R.getNumArguments();
    for (unsigned ArgIdx = 0; ArgIdx < NumArgs; ArgIdx++) {
      if (!NewFuncType.getInput(ArgIdx).isa<MemRefType>())
        continue;
      BlockArgument NewArg = R.insertArgument(
          ArgIdx, NewFuncType.getInput(ArgIdx), rewriter.getUnknownLoc());
      BlockArgument OldArg = R.getArgument(ArgIdx + 1);
      OldArg.replaceAllUsesWith(NewArg);
      R.eraseArgument(ArgIdx + 1);
    }
    rewriter.finalizeRootUpdate(op);
    return success();
  }
};
} // namespace

namespace {
struct ContractionToAffineLoweringPass
    : public PassWrapper<ContractionToAffineLoweringPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ContractionToAffineLoweringPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, func::FuncDialect, memref::MemRefDialect>();
  }
  void runOnOperation() final;
};
} // namespace

void ContractionToAffineLoweringPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addLegalDialect<AffineDialect, BuiltinDialect, arith::ArithDialect,
                         memref::MemRefDialect>();
  target.addIllegalDialect<contraction::ContractionDialect>();
  target.addDynamicallyLegalOp<func::FuncOp>([](func::FuncOp op) {
    return llvm::none_of(op.getResultTypes(),
                         [](Type T) { return T.isa<TensorType>(); });
  });
  target.addDynamicallyLegalOp<func::ReturnOp>([](func::ReturnOp op) {
    return llvm::none_of(op.getOperandTypes(),
                         [](Type T) { return T.isa<TensorType>(); });
  });
  RewritePatternSet patterns(&getContext());
  patterns.add<AddOpLowering, MulOpLowering, IndexOpLowering, ForOpLowering,
               ContractOpLowering, ReturnOpLowering, FuncOpLowering>(
      &getContext());
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::contraction::createLowerToAffinePass() {
  return std::make_unique<ContractionToAffineLoweringPass>();
}
