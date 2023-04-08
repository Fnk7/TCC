#include "TCC/Dialect/TC/IR/TCOps.h"
#include "TCC/Dialect/TC/IR/TCPatterns.h"
#include "TCC/Dialect/TC/Utils/AxesMap.h"
#include "TCC/Dialect/TC/Utils/Tensor.h"
#include "TCC/Dialect/TC/Utils/Utils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::tc {
// PermuteOp: Self PermuteFoldPattern
// Remove PermuteOp without order change
struct PermuteFoldPattern : public OpRewritePattern<PermuteOp> {
  using OpRewritePattern<PermuteOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(PermuteOp op,
                                PatternRewriter &rewriter) const override {
    auto axesMap = op.getAxesMapArray();
    if (isOrderedAxesMap(axesMap)) {
      rewriter.replaceOp(op, op.getInput());
      return success();
    }
    return failure();
  }
};

// PermuteOp: PermuteOp PermuteFoldPattern
// Fold two PermuteOp into one
struct PermutePermuteFoldPattern : public OpRewritePattern<PermuteOp> {
  using OpRewritePattern<PermuteOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(PermuteOp op,
                                PatternRewriter &rewriter) const override {
    auto input = op.getInput();
    if (auto op2 = input.getDefiningOp<PermuteOp>()) {
      auto input2 = op2.getInput();
      auto axesMap1 = op.getAxesMapArray();
      // BUG: 04-04
      // forget '2' in op2.getAxesMapArray
      // 5 hour for debug......
      auto axesMap2 = op2.getAxesMapArray();
      // for (auto &axis : axesMap2) {
      //   axis = axesMap1[axis];
      // }
      utils::applyAxesMap(axesMap1, axesMap2);
      rewriter.replaceOpWithNewOp<PermuteOp>(op, input2, axesMap2);
      return success();
    }
    return failure();
  }
};

// PermuteOp: Self PermuteSymmetryPattern
// Canonicalize axes_map when input is symmetry
struct PermuteSymmetryPattern : public OpRewritePattern<PermuteOp> {
  using OpRewritePattern<PermuteOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(PermuteOp op,
                                PatternRewriter &rewriter) const override {
    auto input = op.getInput();
    auto symmetry = getTensorSymmetry(input);
    if (symmetry.empty())
      return failure();
    auto axesMap = op.getAxesMapArray();
    bool changed = false;
    for (auto [axis1, axis2] : symmetry) {
      assert(axis1 < axis2);
      if (axesMap[axis1] > axesMap[axis2]) {
        changed = true;
        std::swap(axesMap[axis1], axesMap[axis2]);
      }
    }
    if (!changed)
      return failure();
    rewriter.replaceOpWithNewOp<PermuteOp>(op, input, axesMap);
    return success();
  }
};

void populatePermuteSimplifyPatterns(RewritePatternSet &patterns,
                                     MLIRContext *context) {
  patterns.add<PermutePermuteFoldPattern>(context);
  patterns.add<PermuteFoldPattern>(context);
  patterns.add<PermuteSymmetryPattern>(context);
}

void PermuteOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                            MLIRContext *context) {
  populatePermuteSimplifyPatterns(patterns, context);
}
} // namespace mlir::tc
