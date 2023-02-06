#include "TCC/Dialect/TC/IR/TCOps.h"
#include "TCC/Dialect/TC/Utils/Utils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::tc;

#define TC_CANONICALIZE

// AddOp: Self AddReorderPattern
// Reorder input tensors
struct AddReorderPattern : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> inputs = op.getInputs();
    auto newOrder = getNewInputsOrder(inputs);
    if (newOrder.empty())
      return failure();
    auto signsAttr = op.getSigns();
    auto axesMapsAttr = op.getAxesMaps();
    SmallVector<Value> newInputs;
    SmallVector<Attribute> newSigns;
    SmallVector<Attribute> newAxesMaps;
    for (auto idx : newOrder) {
      newInputs.emplace_back(inputs[idx]);
      newSigns.emplace_back(signsAttr[idx]);
      newAxesMaps.emplace_back(axesMapsAttr[idx]);
    }
    auto newSignsAttr = rewriter.getArrayAttr(newSigns);
    auto newAxesMapsAttr = rewriter.getArrayAttr(newAxesMaps);
    rewriter.replaceOpWithNewOp<AddOp>(op, op.getType(), newInputs,
                                       newAxesMapsAttr, newSignsAttr);
    return success();
  }
};

// AddOp: Self AddAttrCanonicalizePattern
// Canonicalize axes maps
struct AddAttrCanonicalizePattern : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    auto dim = getTensorRank(op.getResult());
    auto axesMaps = getAxesMaps(op);
    AxesMapTy inverseMap = canonicalizeAxesMaps(axesMaps, dim);
    if (isOrderedAxesMap(inverseMap)) {
      return failure();
    }
    AxesMapTy permuteMap = inverseAxesMap(inverseMap);
    auto newOp = rewriter.create<AddOp>(op.getLoc(), op.getInputs(), axesMaps,
                                        op.getSignsAttr());
    rewriter.replaceOpWithNewOp<PermuteOp>(op, newOp.getResult(), permuteMap);
    return success();
  }
};

struct AddNegFlodPattern : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    bool change = false;
    auto inputs = op.getInputs();
    auto signsAttr = op.getSigns();
    SmallVector<Value> newInputs;
    SmallVector<bool> newSigns;
    for (auto [input, signAttr] : llvm::zip(inputs, signsAttr)) {
      auto sign = signAttr.cast<BoolAttr>().getValue();
      if (auto negOp = input.getDefiningOp<NegOp>()) {
        change = true;
        newInputs.emplace_back(negOp.getInput());
        newSigns.emplace_back(!sign);
      } else {
        newInputs.emplace_back(input);
        newSigns.emplace_back(sign);
      }
    }
    if (!change) {
      return failure();
    }
    auto newSignsAttr = rewriter.getBoolArrayAttr(newSigns);
    rewriter.replaceOpWithNewOp<AddOp>(op, op.getType(), newInputs,
                                       op.getAxesMaps(), newSignsAttr);
    return success();
  }
};

// AddOp: AddOp AddAddFoldPattern
// Fold AddOp into AddOp
struct AddAddFoldPattern : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    auto inputs = op.getInputs();
    auto axesMaps = getAxesMaps(op);
    auto signs = getSigns(op);
    SmallVector<Value> newInputs;
    AxesMapsTy newAxesMaps;
    SignsTy newSigns;
    bool changed = false;
    for (auto [input, axesMap, sign] : llvm::zip(inputs, axesMaps, signs)) {
      if (auto op2 = input.getDefiningOp<AddOp>()) {
        changed = true;
        auto inputs2 = op2.getInputs();
        auto axesMaps2 = getAxesMaps(op2);
        auto signs2 = getSigns(op2);
        for (auto [input2, axesMap2, sign2] :
             llvm::zip(inputs2, axesMaps2, signs2)) {
          for (auto &axis : axesMap2) {
            axis = axesMap[axis];
          }
          newInputs.emplace_back(input2);
          newAxesMaps.emplace_back(axesMap2);
          newSigns.emplace_back(!(sign ^ sign2));
        }
      } else {
        newInputs.emplace_back(input);
        newAxesMaps.emplace_back(axesMap);
        newSigns.emplace_back(sign);
      }
    }
    if (!changed) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<AddOp>(op, newInputs, newAxesMaps, newSigns);
    return success();
  }
};

// AddOp: PermuteOp AddPermuteFoldPattern
// Fold PermuteOp into AddOp
struct AddPermuteFoldPattern : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    bool changed = false;
    auto inputs = op.getInputs();
    auto axesMaps = getAxesMaps(op);
    SmallVector<Value> newInputs;
    AxesMapsTy newAxesMaps;
    for (auto [input, axesMap] : llvm::zip(inputs, axesMaps)) {
      if (auto permuteOp = input.getDefiningOp<PermuteOp>()) {
        changed = true;
        auto permuteMap = getAxesMap(permuteOp);
        assert(axesMap.size() == permuteMap.size());
        // Permute Input  [s0,s1,s2,s3], b0~b3
        // Permute Result [s3,s0,s2,s1], a0~a3
        // Permute Map [1,3,2,0]; a = amap[b]; sizeIDa[amap[b]] = sizeIDb[b]
        // AxesMap     [i0,i1,i2,i3], i1 -> s0
        // new AxesMap [i1,i3,i2,i0], i1 -> s0
        AxesMapTy newAxesMap;
        newAxesMap.reserve(permuteMap.size());
        for (auto axis : permuteMap) {
          newAxesMap.emplace_back(axesMap[axis]);
        }
        newInputs.emplace_back(permuteOp.getInput());
        newAxesMaps.emplace_back(newAxesMap);
      } else {
        newInputs.emplace_back(input);
        newAxesMaps.emplace_back(axesMap);
      }
    }
    if (!changed) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<AddOp>(op, newInputs, newAxesMaps,
                                       op.getSignsAttr());
    return success();
  }
};

// AddOp
// Self: AddReorderPattern, AddAttrCanonicalizePattern
// Mul:
// Add: AddAddFoldPattern
// Contract: Deleted
// Permute: AddPermuteFoldPattern
// Others
// AddConstantFoldPattern
// AddNegFoldPattern
void AddOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                        MLIRContext *context) {
#ifdef TC_CANONICALIZE_V0
  patterns.add<AddNegFlodPattern>(context);
  // patterns.add<AddPermuteFoldPattern>(context);
  patterns.add<AddAddFoldPattern>(context);
  patterns.add<AddReorderPattern>(context);
  patterns.add<AddAttrCanonicalizePattern>(context);
#elif defined(TC_CANONICALIZE)
  patterns.add<AddReorderPattern>(context);
  patterns.add<AddAttrCanonicalizePattern>(context);
  patterns.add<AddAddFoldPattern>(context);
  patterns.add<AddPermuteFoldPattern>(context);
#endif
}

struct MulNegSwapPattern : public OpRewritePattern<MulOp> {
  using OpRewritePattern<MulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {
    int negCount = 0;
    SmallVector<Value> newInputs;
    for (auto input : op.getInputs()) {
      if (auto negOp = input.getDefiningOp<NegOp>()) {
        negCount++;
        newInputs.emplace_back(negOp.getInput());
      } else {
        newInputs.emplace_back(input);
      }
    }
    if (negCount == 0) {
      return failure();
    }
    if (negCount % 2 == 0) {
      rewriter.replaceOpWithNewOp<MulOp>(op, op.getType(), newInputs,
                                         op.getAxesMaps());
    } else {
      auto newMulOp = rewriter.create<MulOp>(op.getLoc(), op.getType(),
                                             newInputs, op.getAxesMaps());
      rewriter.replaceOpWithNewOp<NegOp>(op, op.getType(), newMulOp);
    }
    return success();
  }
};

// MulOp: Self MulReorderPattern
// Reorder input tensors
struct MulReorderPattern : public OpRewritePattern<MulOp> {
  using OpRewritePattern<MulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> inputs = op.getInputs();
    auto newOrder = getNewInputsOrder(inputs);
    if (newOrder.empty())
      return failure();
    auto axesMapsAttr = op.getAxesMaps();
    SmallVector<Value> newInputs;
    SmallVector<Attribute> newAxesMaps;
    for (auto idx : newOrder) {
      newInputs.emplace_back(inputs[idx]);
      newAxesMaps.emplace_back(axesMapsAttr[idx]);
    }
    auto newAxesMapsAttr = rewriter.getArrayAttr(newAxesMaps);
    rewriter.replaceOpWithNewOp<MulOp>(op, op.getType(), newInputs,
                                       newAxesMapsAttr);
    return success();
  }
};

// MulOp: Self MulAttrCanonicalizePattern
// Canonicalize axes maps
struct MulAttrCanonicalizePattern : public OpRewritePattern<MulOp> {
  using OpRewritePattern<MulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {
    auto dim = getTensorRank(op.getResult());
    auto axesMaps = getAxesMaps(op);
    AxesMapTy inverseMap = canonicalizeAxesMaps(axesMaps, dim);
    if (isOrderedAxesMap(inverseMap)) {
      return failure();
    }
    AxesMapTy permuteMap = inverseAxesMap(inverseMap);
    auto newOp = rewriter.create<MulOp>(op.getLoc(), op.getInputs(), axesMaps);
    rewriter.replaceOpWithNewOp<PermuteOp>(op, newOp.getResult(), permuteMap);
    return success();
  }
};

// MulOp: MulOp MulMulFoldPattern
// Make MulOp not use MulOp as input
struct MulMulFoldPattern : public OpRewritePattern<MulOp> {
  using OpRewritePattern<MulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {
    auto inputs = op.getInputs();
    auto axesMaps = getAxesMaps(op);
    SmallVector<Value> newInputs;
    AxesMapsTy newAxesMaps;
    bool changed = false;
    for (auto [input, axesMap] : llvm::zip(inputs, axesMaps)) {
      if (auto op2 = input.getDefiningOp<MulOp>()) {
        changed = true;
        auto inputs2 = op2.getInputs();
        auto axesMaps2 = getAxesMaps(op2);
        for (auto [input2, axesMap2] : llvm::zip(inputs2, axesMaps2)) {
          for (auto &axis : axesMap2) {
            axis = axesMap[axis];
          }
          newInputs.emplace_back(input2);
          newAxesMaps.emplace_back(axesMap2);
        }
      } else {
        newInputs.emplace_back(input);
        newAxesMaps.emplace_back(axesMap);
      }
    }
    if (!changed) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<MulOp>(op, newInputs, newAxesMaps);
    return success();
  }
};

// MulOp: AddOp MulAddSwapPattern
// Make MulOp not use AddOp as input
// Only support elementwise add
struct MulAddSwapPattern : public OpRewritePattern<MulOp> {
  using OpRewritePattern<MulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {
    auto inputs = op.getInputs();
    auto axesMaps = getAxesMaps(op);
    SmallVector<Value> newInputs;
    AxesMapsTy newAxesMaps;
    int64_t addOpIdx = -1;
    for (auto e : llvm::enumerate(inputs)) {
      if (addOpIdx < 0 && e.value().getDefiningOp<AddOp>()) {
        addOpIdx = e.index();
      } else {
        newInputs.emplace_back(e.value());
        newAxesMaps.emplace_back(axesMaps[e.index()]);
      }
    }
    if (addOpIdx < 0) {
      return failure();
    }
    auto addOp = inputs[addOpIdx].getDefiningOp<AddOp>();
    auto addInputs = addOp.getInputs();
    auto addAxesMaps = getAxesMaps(addOp);
    auto rank = addAxesMaps[0].size();
    for (auto &axesMap : addAxesMaps) {
      // Only support elementwise add
      if (axesMap.size() != rank) {
        return failure();
      }
    }
    auto &axesMap = axesMaps[addOpIdx];
    SmallVector<Value> newAddInputs;
    AxesMapsTy newAddAxesMaps;
    for (auto [addInput, addAxesMap] : llvm::zip(addInputs, addAxesMaps)) {
      for (auto &axis : addAxesMap) {
        axis = axesMap[axis];
      }
      newInputs.emplace_back(addInput);
      newAxesMaps.emplace_back(addAxesMap);
      auto newOp = rewriter.create<MulOp>(op.getLoc(), newInputs, newAxesMaps);
      newInputs.pop_back();
      newAxesMaps.pop_back();
      newAddInputs.emplace_back(newOp.getResult());
    }
    auto dim = getTensorRank(op.getResult());
    auto r = llvm::seq<int64_t>(0, dim);
    for (auto _ : llvm::seq<int64_t>(0, newAddInputs.size())) {
      (void)_;
      newAddAxesMaps.emplace_back(AxesMapTy(r.begin(), r.end()));
    }
    rewriter.replaceOpWithNewOp<AddOp>(op, newAddInputs, newAddAxesMaps,
                                       addOp.getSignsAttr());
    return success();
  }
};

// MulOp: ContractOp MulContractSwapPattern
// Make MulOp not use ContractOp as input
// a(i) * sum((j), b(j,k)) * sum((l), c(l,i))
// sum((j), a(i) * b(j,k) * sum((l), c(l,i)))
// sum((j), sum((l), a(i) * b(j,k) * c(l,i)))
struct MulContractSwapPattern : public OpRewritePattern<MulOp> {
  using OpRewritePattern<MulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {
    auto dim = getTensorRank(op.getResult());
    auto axesMaps = getAxesMaps(op);
    auto inputs = op.getInputs();
    assert(inputs.size() >= 2);
    SmallVector<Value> newInputs;
    AxesMapsTy newAxesMaps;
    AxesTy newAxes;
    auto newAxis = dim;
    bool changed = false;
    for (auto [input, axesMap] : llvm::zip(inputs, axesMaps)) {
      if (auto contractOp = input.getDefiningOp<ContractOp>()) {
        changed = true;
        auto axes = getAxes(contractOp);
        auto i2oAxisMap = getContractOpInput2OutputAxisMap(
            axes, axesMap.size() + axes.size());
        for (auto &newAxis : i2oAxisMap) {
          if (newAxis > 0)
            newAxis = axesMap[newAxis];
        }
        for (auto axis : axes) {
          i2oAxisMap[axis] = newAxis;
          newAxes.emplace_back(newAxis);
          newAxis++;
        }
        newInputs.emplace_back(contractOp.getInput());
        newAxesMaps.emplace_back(i2oAxisMap);
      } else {
        newInputs.emplace_back(input);
        newAxesMaps.emplace_back(axesMap);
      }
    }
    if (!changed) {
      return failure();
    }
    auto newOp = rewriter.create<MulOp>(op.getLoc(), newInputs, newAxesMaps);
    rewriter.replaceOpWithNewOp<ContractOp>(op, (Value)newOp, newAxes);
    return success();
  }
};

// MulOp: PermuteOp MulPermuteFoldPattern
// Fold PermuteOp into MulOp
struct MulPermuteFoldPattern : public OpRewritePattern<MulOp> {
  using OpRewritePattern<MulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {
    bool changed = false;
    auto inputs = op.getInputs();
    auto axesMaps = getAxesMaps(op);
    SmallVector<Value> newInputs;
    AxesMapsTy newAxesMaps;
    for (auto [input, axesMap] : llvm::zip(inputs, axesMaps)) {
      if (auto permuteOp = input.getDefiningOp<PermuteOp>()) {
        changed = true;
        auto permuteMap = getAxesMap(permuteOp);
        assert(axesMap.size() == permuteMap.size());
        // Permute Input  [s0,s1,s2,s3], b0~b3
        // Permute Result [s3,s0,s2,s1], a0~a3
        // Permute Map [1,3,2,0]; a = amap[b]; sizeIDa[amap[b]] = sizeIDb[b]
        // AxesMap     [i0,i1,i2,i3], i1 -> s0
        // new AxesMap [i1,i3,i2,i0], i1 -> s0
        AxesMapTy newAxesMap;
        newAxesMap.reserve(permuteMap.size());
        for (auto axis : permuteMap) {
          newAxesMap.emplace_back(axesMap[axis]);
        }
        newInputs.emplace_back(permuteOp.getInput());
        newAxesMaps.emplace_back(newAxesMap);
      } else {
        newInputs.emplace_back(input);
        newAxesMaps.emplace_back(axesMap);
      }
    }
    if (!changed) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<MulOp>(op, newInputs, newAxesMaps);
    return success();
  }
};

// MulOp
// Self: MulReorderPattern, MulAttrCanonicalizePattern
// Mul: MulMulFoldPattern
// Add: MulAddSwapPattern
// Contract: MulContractSwapPattern
// Permute: MulPermuteFoldPattern
// Others
// MulConstantFoldPattern
// MulNegSwapPattern
void MulOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                        MLIRContext *context) {
#ifdef TC_CANONICALIZE
  patterns.add<MulReorderPattern>(context);
  patterns.add<MulAttrCanonicalizePattern>(context);
  patterns.add<MulMulFoldPattern>(context);
  patterns.add<MulAddSwapPattern>(context);
  patterns.add<MulContractSwapPattern>(context);
  patterns.add<MulPermuteFoldPattern>(context);
#endif
}

// PermuteOp: Self PermuteFoldPattern
// Remove PermuteOp without order change
struct PermuteFoldPattern : public OpRewritePattern<PermuteOp> {
  using OpRewritePattern<PermuteOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(PermuteOp op,
                                PatternRewriter &rewriter) const override {
    auto axesMap = getAxesMap(op);
    if (isOrderedAxesMap(axesMap)) {
      rewriter.replaceOp(op, op.getInput());
      return success();
    }
    return failure();
  }
};

// PermuteOp: PermuteOp PermuteFoldPattern
// Fold two PermuteOp
struct PermutePermuteFoldPattern : public OpRewritePattern<PermuteOp> {
  using OpRewritePattern<PermuteOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(PermuteOp op,
                                PatternRewriter &rewriter) const override {
    auto input = op.getInput();
    if (auto op2 = input.getDefiningOp<PermuteOp>()) {
      auto input2 = op2.getInput();
      auto axesMap1 = getAxesMap(op);
      auto axesMap2 = getAxesMap(op2);
      for (auto &axis : axesMap2) {
        axis = axesMap1[axis];
      }
      rewriter.replaceOpWithNewOp<PermuteOp>(op, input2, axesMap2);
      return success();
    }
    return failure();
  }
};

// PermuteOp
// Self: PermuteFoldPattern
// Mul, Add, Contract:
// Permute: PermutePermuteFoldPattern
// Others
void PermuteOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                            MLIRContext *context) {
#ifdef TC_CANONICALIZE
  patterns.add<PermuteFoldPattern>(context);
  patterns.add<PermutePermuteFoldPattern>(context);
#endif
}

struct NegNegFoldPattern : public OpRewritePattern<NegOp> {
  using OpRewritePattern<NegOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(NegOp op,
                                PatternRewriter &rewriter) const override {
    auto input = op.getInput();
    if (auto op2 = input.getDefiningOp<NegOp>()) {
      rewriter.replaceOp(op, op2.getInput());
      return success();
    }
    return failure();
  }
};

// NegOp
// Mul, Add, Contract:
// Permute: NegPermuteSwapPattern
// Others
// NegNegFoldPattern
void NegOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                        MLIRContext *context) {
#ifdef TC_CANONICALIZE_V0
  patterns.add<NegNegFoldPattern>(context);
#endif
}

// ContractOp: AddOp ContractAddSwapPattern
// Make ContractOp not use AddOp as input
// sum((j),A(i) + B(j)), can not handle
// sum((j),A(i)) + sum((j),B(j)) = size(j) * A(i) + sum((j),B(j))
struct ContractAddSwapPattern : public OpRewritePattern<ContractOp> {
  using OpRewritePattern<ContractOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ContractOp op,
                                PatternRewriter &rewriter) const override {
    auto input = op.getInput();
    if (auto addOp = input.getDefiningOp<AddOp>()) {
      auto axesMaps = getAxesMaps(addOp);
      auto rank = axesMaps[0].size();
      for (auto &axesMap : axesMaps) {
        // can not handle AddOp inputs ranks are not same
        if (axesMap.size() != rank)
          return failure();
      }
      auto axes = getAxes(op);
      auto i2oAxisMap = getContractOpInput2OutputAxisMap(axes, rank);
      SmallVector<Value> newInputs;
      AxesMapsTy newAxesMaps;
      for (auto [input, axesMap] : llvm::zip(addOp.getInputs(), axesMaps)) {
        AxesTy newAxes;
        AxesMapTy newAxesMap;
        for (auto e : llvm::enumerate(axesMap)) {
          auto oldAxis = e.value();
          auto newAxis = i2oAxisMap[oldAxis];
          if (newAxis >= 0) {
            newAxesMap.emplace_back(newAxis);
          } else {
            newAxes.emplace_back(e.index());
          }
        }
        auto contractOp =
            rewriter.create<ContractOp>(op.getLoc(), input, newAxes);
        newInputs.emplace_back(contractOp);
        newAxesMaps.emplace_back(newAxesMap);
      }
      rewriter.replaceOpWithNewOp<AddOp>(op, newInputs, newAxesMaps,
                                         addOp.getSignsAttr());
      return success();
    }
    return failure();
  }
};

// ContractOp: ContractOp ContractContractFoldPattern
// Make ContractOp not use ContractOp as input
struct ContractContractFoldPattern : public OpRewritePattern<ContractOp> {
  using OpRewritePattern<ContractOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ContractOp op,
                                PatternRewriter &rewriter) const override {
    auto input = op.getInput();
    if (auto op2 = input.getDefiningOp<ContractOp>()) {
      auto axes1 = getAxes(op);
      auto axes2 = getAxes(op2);
      for (auto axis2 : axes2) {
        for (auto &axis1 : llvm::reverse(axes1)) {
          if (axis1 < axis2) {
            break;
          }
          axis1++;
        }
      }
      axes2.append(axes1);
      rewriter.replaceOpWithNewOp<ContractOp>(op, op2.getInput(), axes2);
      return success();
    }
    return failure();
  }
};

// ContractOp: PermuteOp ContractPermuteSwapPattern
// Make ContractOp not use PermuteOp as input
struct ContractPermuteSwapPattern : public OpRewritePattern<ContractOp> {
  using OpRewritePattern<ContractOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ContractOp op,
                                PatternRewriter &rewriter) const override {
    auto input = op.getInput();
    if (auto permuteOp = input.getDefiningOp<PermuteOp>()) {
      // [0,2], axes
      // [1,3,2,0], axesMap
      // [3,2], newAxes
      // [x,0,x,1], i2oAxisMap
      // [0,1], newAxesMap
      auto axes = getAxes(op);
      auto axesMap = getAxesMap(permuteOp);
      AxesTy newAxes;
      for (auto axis : axes) {
        for (auto e : llvm::enumerate(axesMap)) {
          if (axis == e.value())
            newAxes.emplace_back(e.index());
        }
      }
      auto i2oAxisMap = getContractOpInput2OutputAxisMap(axes, axesMap.size());
      AxesMapTy newAxesMap;
      for (auto axis : axesMap) {
        auto newAxis = i2oAxisMap[axis];
        if (newAxis >= 0) {
          newAxesMap.emplace_back(newAxis);
        }
      }
      auto newOp = rewriter.create<ContractOp>(op.getLoc(),
                                               permuteOp.getInput(), newAxes);
      rewriter.replaceOpWithNewOp<PermuteOp>(op, (Value)newOp, newAxesMap);
      return success();
    }
    return failure();
  }
};

// ContractOp
// Mul:
// Add: ContractAddSwapPattern
// Contract: ContractContractFoldPattern
// Permute: ContractPermuteSwapPattern
// Others
// ContractNegSwapPattern
void ContractOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                             MLIRContext *context) {
#ifdef TC_CANONICALIZE
  patterns.add<ContractAddSwapPattern>(context);
  patterns.add<ContractContractFoldPattern>(context);
  patterns.add<ContractPermuteSwapPattern>(context);
#endif
}
