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
static SmallVector<int64_t> unpackIntegerArrayAttr(ArrayAttr Attr) {
  SmallVector<int64_t> Sizes;
  for (auto &I64 : Attr) {
    Sizes.emplace_back(I64.cast<IntegerAttr>().getValue().getSExtValue());
  }
  return Sizes;
}
static SmallVector<int64_t> getTensorSizes(RankedTensorType RTT) {
  auto Attr = RTT.getEncoding().cast<ArrayAttr>();
  return unpackIntegerArrayAttr(Attr);
}
static SmallVector<int64_t> getTensorSizes(Value V) {
  auto RTT = V.getType().cast<RankedTensorType>();
  return getTensorSizes(RTT);
}
struct NewBinaryOpLowering2
    : public OpConversionPattern<contraction::NewAddOp> {
  using OpConversionPattern<contraction::NewAddOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(contraction::NewAddOp op,
                  typename contraction::NewAddOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto Loc = op.getLoc();
    auto InType = op.getResult().getType().cast<RankedTensorType>();
    auto OutType = MemRefType::get(InType.getShape(), InType.getElementType());
    auto Sizes = getTensorSizes(InType);
    SmallVector<Value> SizeValues;
    auto &B = *(op->getBlock());
    for (auto &Sz : Sizes) {
      SizeValues.emplace_back(B.getArgument(Sz));
    }
    auto Alloc = rewriter.create<memref::AllocOp>(Loc, OutType, SizeValues);
    Alloc->moveBefore(&(B.front()));
    bool NeedDealloc = true;
    for (auto *U : op.getResult().getUsers()) {
      if (dyn_cast<func::ReturnOp>(U))
        NeedDealloc = false;
    }
    if (NeedDealloc) {
      auto Dealloc = rewriter.create<memref::DeallocOp>(Loc, Alloc);
      Dealloc->moveBefore(&(B.back()));
    }
    Value ZeroIndexVar = rewriter.create<arith::ConstantIndexOp>(Loc, 0);
    SmallVector<Value> InductionVars;
    Block *InnerMostBody;
    for (auto SzValue : SizeValues) {
      auto AffineFor = rewriter.create<AffineForOp>(
          Loc, ZeroIndexVar, rewriter.getSymbolIdentityMap(), SzValue,
          rewriter.getSymbolIdentityMap(), 1);
      InnerMostBody = AffineFor.getBody();
      InductionVars.emplace_back(AffineFor.getInductionVar());
      rewriter.setInsertionPointToStart(InnerMostBody);
    }
    SmallVector<Value> Loads;
    auto AxesMapsAttr = op.getAxesMaps();
    for (size_t Operand = 0; Operand < AxesMapsAttr.size(); Operand++) {
      Value ToLoad = adaptor.getTensors()[Operand];
      SmallVector<Value> Indexes;
      auto AxesMapAttr = AxesMapsAttr[Operand].cast<ArrayAttr>();
      auto AxesMap = unpackIntegerArrayAttr(AxesMapAttr);
      for (auto Axis : AxesMap) {
        Indexes.emplace_back(InductionVars[Axis]);
      }
      Value Loaded = rewriter.create<memref::LoadOp>(Loc, ToLoad, Indexes);
      Loads.emplace_back(Loaded);
    }
    Value ToStore = Loads[0];
    for (size_t Operand = 1; Operand < Loads.size(); Operand++) {
      ToStore = rewriter.create<arith::AddFOp>(Loc, ToStore, Loads[Operand]);
    }
    rewriter.create<memref::StoreOp>(Loc, ToStore, Alloc, InductionVars);
    rewriter.replaceOp(op, Alloc->getResult(0));
    return success();
  }
};

template <typename BinaryOp, typename LoweredBinaryOp>
struct NewBinaryOpLowering : public OpConversionPattern<BinaryOp> {
  using OpConversionPattern<BinaryOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BinaryOp op, typename BinaryOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto Loc = op.getLoc();
    auto InType = cast<RankedTensorType>(op.getResult().getType());
    auto OutType = MemRefType::get(InType.getShape(), InType.getElementType());
    auto Sizes = getTensorSizes(InType);
    SmallVector<Value> SizeValues;
    auto &B = *(op->getBlock());
    for (auto &Sz : Sizes) {
      SizeValues.emplace_back(B.getArgument(Sz));
    }
    auto Alloc = rewriter.create<memref::AllocOp>(Loc, OutType, SizeValues);
    Alloc->moveBefore(&(B.front()));
    bool NeedDealloc = true;
    for (auto *U : op.getResult().getUsers()) {
      if (dyn_cast<func::ReturnOp>(U))
        NeedDealloc = false;
    }
    if (NeedDealloc) {
      auto Dealloc = rewriter.create<memref::DeallocOp>(Loc, Alloc);
      Dealloc->moveBefore(&(B.back()));
    }
    Value ZeroIndexVar = rewriter.create<arith::ConstantIndexOp>(Loc, 0);
    SmallVector<Value> InductionVars;
    Block *InnerMostBody;
    for (auto SzValue : SizeValues) {
      auto AffineFor = rewriter.create<AffineForOp>(
          Loc, ZeroIndexVar, rewriter.getSymbolIdentityMap(), SzValue,
          rewriter.getSymbolIdentityMap(), 1);
      InnerMostBody = AffineFor.getBody();
      InductionVars.emplace_back(AffineFor.getInductionVar());
      rewriter.setInsertionPointToStart(InnerMostBody);
    }
    SmallVector<Value> Loads;
    auto AxesMapsAttr = op.getAxesMaps();
    for (size_t Operand = 0; Operand < AxesMapsAttr.size(); Operand++) {
      Value LoadFrom = adaptor.getTensors()[Operand];
      SmallVector<Value> Indexes;
      auto AxesMapAttr = cast<ArrayAttr>(AxesMapsAttr[Operand]);
      auto AxesMap = unpackIntegerArrayAttr(AxesMapAttr);
      for (auto Axis : AxesMap) {
        Indexes.emplace_back(InductionVars[Axis]);
      }
      Value Loaded = rewriter.create<memref::LoadOp>(Loc, LoadFrom, Indexes);
      Loads.emplace_back(Loaded);
    }
    Value ToStore = Loads[0];
    for (size_t Operand = 1; Operand < Loads.size(); Operand++) {
      ToStore = rewriter.create<LoweredBinaryOp>(Loc, ToStore, Loads[Operand]);
    }
    rewriter.create<memref::StoreOp>(Loc, ToStore, Alloc, InductionVars);
    rewriter.replaceOp(op, Alloc->getResults());
    return success();
  }
};

using NewAddOpLowering =
    NewBinaryOpLowering<contraction::NewAddOp, arith::AddFOp>;
using NewMulOpLowering =
    NewBinaryOpLowering<contraction::NewMulOp, arith::MulFOp>;

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

struct NewReturnOpLowering : public OpConversionPattern<func::ReturnOp> {
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
    auto Offset = NISAttr.getValue().getSExtValue();
    assert(WriteBacks.size() == ReturnValues.size());
    SmallVector<Value> NewReturnValues;
    for (auto [WriteBackIdx, ReturnValue] :
         llvm::zip(WriteBacks, ReturnValues)) {
      if (WriteBackIdx < 0) {
        NewReturnValues.emplace_back(ReturnValue);
      } else {
        auto InputValue = B.getArgument(WriteBackIdx + Offset);
        auto CopyTo = (*(InputValue.getUsers().begin()))->getResult(0);
        auto Copy = rewriter.create<memref::CopyOp>(Loc, ReturnValue, CopyTo);
        Copy->moveBefore(&B.back());
        auto Dealloc = rewriter.create<memref::DeallocOp>(Loc, ReturnValue);
        Dealloc->moveBefore(&B.back());
      }
    }
    rewriter.updateRootInPlace(op, [&] { op->setOperands(NewReturnValues); });
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
    rewriter.startRootUpdate(op);
    SmallVector<Type> NewInputTypes;
    SmallVector<Type> NewResultTypes;
    for (unsigned ArgIdx = 0; ArgIdx < InputTypes.size(); ArgIdx++) {
      auto T = InputTypes[ArgIdx];
      if (auto TT = T.dyn_cast<TensorType>()) {
        NewInputTypes.emplace_back(
            MemRefType::get(TT.getShape(), TT.getElementType()));
        op.getRegion().getArgument(ArgIdx).setType(NewInputTypes.back());
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
    op.setType(NewFuncType);
    rewriter.finalizeRootUpdate(op);
    return success();
  }
};

struct NewFuncOpLowering : public OpConversionPattern<func::FuncOp> {
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
    auto WriteBackAttr = FuncAttr.get("wirtebacks").cast<ArrayAttr>();
    auto WriteBacks = unpackIntegerArrayAttr(WriteBackAttr);
    SmallVector<Type> NewInputTypes(NumInputSizes, rewriter.getIndexType());
    SmallVector<Type> NewResultTypes;
    auto &R = op.getRegion();
    rewriter.setInsertionPointToStart(&(R.front()));
    Value ZeroOffset = rewriter.create<arith::ConstantIndexOp>(Loc, 0);
    unsigned ArgIdx = 0;
    for (; ArgIdx < NumInputSizes; ArgIdx++) {
      R.insertArgument(ArgIdx, rewriter.getIndexType(), Loc);
    }
    for (auto T : InputTypes) {
      assert(T.isa<RankedTensorType>());
      auto RTT = T.cast<RankedTensorType>();
      auto ViewOpType = MemRefType::get(RTT.getShape(), RTT.getElementType());
      auto Arg = R.getArgument(ArgIdx++);
      auto ArgType = MemRefType::get({(int64_t)-1}, rewriter.getI8Type());
      NewInputTypes.emplace_back(ArgType);
      Arg.setType(ArgType);
      auto Sizes = getTensorSizes(RTT);
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
struct PermuteOpLowering : public OpConversionPattern<contraction::PermuteOp> {
  using OpConversionPattern<contraction::PermuteOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(contraction::PermuteOp op,
                  contraction::PermuteOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto Loc = op.getLoc();
    auto InType = op.getResult().getType().cast<RankedTensorType>();
    auto OutType = MemRefType::get(InType.getShape(), InType.getElementType());
    auto Sizes = getTensorSizes(InType);
    SmallVector<Value> SizeValues;
    auto &B = *(op->getBlock());
    for (auto &Sz : Sizes) {
      SizeValues.emplace_back(B.getArgument(Sz));
    }
    auto Alloc = rewriter.create<memref::AllocOp>(Loc, OutType, SizeValues);
    Alloc->moveBefore(&(B.front()));
    bool NeedDealloc = true;
    for (auto *U : op.getResult().getUsers()) {
      if (dyn_cast<func::ReturnOp>(U))
        NeedDealloc = false;
    }
    if (NeedDealloc) {
      auto Dealloc = rewriter.create<memref::DeallocOp>(Loc, Alloc);
      Dealloc->moveBefore(&(B.back()));
    }
    Value ZeroIndexVar = rewriter.create<arith::ConstantIndexOp>(Loc, 0);
    SmallVector<Value> InductionVars;
    Block *InnerMostBody;
    for (auto SzValue : SizeValues) {
      auto AffineFor = rewriter.create<AffineForOp>(
          Loc, ZeroIndexVar, rewriter.getSymbolIdentityMap(), SzValue,
          rewriter.getSymbolIdentityMap(), 1);
      InnerMostBody = AffineFor.getBody();
      InductionVars.emplace_back(AffineFor.getInductionVar());
      rewriter.setInsertionPointToStart(InnerMostBody);
    }
    Value LoadFrom = adaptor.getTensor();
    SmallVector<Value> Indexes;
    auto AxesMapAttr = op.getAxesMap();
    auto AxesMap = unpackIntegerArrayAttr(AxesMapAttr);
    for (auto Axis : AxesMap) {
      Indexes.emplace_back(InductionVars[Axis]);
    }
    Value ToStore = rewriter.create<memref::LoadOp>(Loc, LoadFrom, Indexes);
    rewriter.create<memref::StoreOp>(Loc, ToStore, Alloc, InductionVars);
    rewriter.replaceOp(op, Alloc->getResults());
    return success();
  }
};
struct NewContractOpLowering
    : public OpConversionPattern<contraction::NewContractOp> {
  using OpConversionPattern<contraction::NewContractOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(contraction::NewContractOp op,
                  contraction::NewContractOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto Loc = op.getLoc();
    auto InType = op.getResult().getType().cast<RankedTensorType>();
    auto OutType = MemRefType::get(InType.getShape(), InType.getElementType());
    auto Sizes = getTensorSizes(InType);
    SmallVector<Value> SizeValues;
    auto &B = *(op->getBlock());
    for (auto &Sz : Sizes) {
      SizeValues.emplace_back(B.getArgument(Sz));
    }
    auto Alloc = rewriter.create<memref::AllocOp>(Loc, OutType, SizeValues);
    Alloc->moveBefore(&(B.front()));
    bool NeedDealloc = true;
    for (auto *U : op.getResult().getUsers()) {
      if (dyn_cast<func::ReturnOp>(U))
        NeedDealloc = false;
    }
    if (NeedDealloc) {
      auto Dealloc = rewriter.create<memref::DeallocOp>(Loc, Alloc);
      Dealloc->moveBefore(&(B.back()));
    }
    Value ZeroIndexVar = rewriter.create<arith::ConstantIndexOp>(Loc, 0);
    SmallVector<Value> InductionVars;
    Block *InnerMostBody;
    for (auto SzValue : SizeValues) {
      auto AffineFor = rewriter.create<AffineForOp>(
          Loc, ZeroIndexVar, rewriter.getSymbolIdentityMap(), SzValue,
          rewriter.getSymbolIdentityMap(), 1);
      InnerMostBody = AffineFor.getBody();
      InductionVars.emplace_back(AffineFor.getInductionVar());
      rewriter.setInsertionPointToStart(InnerMostBody);
    }
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
      InnerMostBody = AffineFor.getBody();
      InductionVars2.emplace_back(AffineFor.getInductionVar());
      ReductionVar = AffineFor.getRegionIterArgs()[0];
      rewriter.setInsertionPointToStart(InnerMostBody);
    }
    Value LoadFrom = adaptor.getTensor();
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
    rewriter.replaceOp(op, Alloc->getResult(0));
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

struct ContractionLoweringPass
    : public PassWrapper<ContractionLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ContractionLoweringPass)

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

void ContractionLoweringPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addLegalDialect<AffineDialect, BuiltinDialect, arith::ArithDialect,
                         memref::MemRefDialect>();
  target.addIllegalDialect<contraction::ContractionDialect>();
  target.addDynamicallyLegalOp<func::FuncOp>([](func::FuncOp op) {
    return llvm::none_of(op.getResultTypes(),
                         [](Type T) { return T.isa<RankedTensorType>(); });
  });
  target.addDynamicallyLegalOp<func::ReturnOp>([](func::ReturnOp op) {
    return llvm::none_of(op.getOperandTypes(),
                         [](Type T) { return T.isa<RankedTensorType>(); });
  });
  RewritePatternSet patterns(&getContext());
  patterns.add<NewFuncOpLowering, NewReturnOpLowering, NewAddOpLowering,
               NewMulOpLowering, PermuteOpLowering, NewContractOpLowering>(
      &getContext());
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::contraction::createLowerToAffinePass() {
  return std::make_unique<ContractionLoweringPass>();
}
