#include "TCC/Dialect/TC/IR/TCOps.h"
#include "TCC/Dialect/TC/Transforms/Passes.h"
#include "TCC/Dialect/TC/Utils/Builder.h"
#include "TCC/Dialect/TC/Utils/Utils.h"
#include "TCC/Utils/InputPostion.h"

#include "mlir/IR/PatternMatch.h"

namespace mlir::tc {
struct GroupElementwiseMulPattern : public OpRewritePattern<MulOp> {
  using OpRewritePattern<MulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {
    auto inputs = op.getInputs();
    if (inputs.size() <= 2)
      return failure();
    auto axesMaps = op.getAxesMaps();
    DenseMap<AxesMapTy, PositionsTy> groups;
    for (auto [pos, axesMap] : llvm::enumerate(axesMaps)) {
      auto key = axesMap;
      llvm::sort(key);
      groups[key].emplace_back(pos);
    }
    // Already is elementwise mul or not contains elementwise mul
    if (groups.size() == 1 || groups.size() == axesMaps.size())
      return failure();
    SmallVector<Value> newInputs;
    AxesMapsTy newAxesMaps;
    for (auto [_, group] : groups) {
      SmallVector<Value> subInputs;
      AxesMapsTy subAxesMaps;
      for (auto pos : group) {
        subInputs.emplace_back(inputs[pos]);
        subAxesMaps.emplace_back(axesMaps[pos]);
      }
      auto [input, axesMap] =
          utils::createMulOp(rewriter, subInputs, subAxesMaps);
      newInputs.emplace_back(input);
      newAxesMaps.emplace_back(axesMap);
    }
    rewriter.replaceOpWithNewOp<MulOp>(op, newInputs, newAxesMaps);
    return success();
  }
};

void populateGroupElementewiseMulPattern(RewritePatternSet &patterns,
                                         MLIRContext *context) {
  patterns.add<GroupElementwiseMulPattern>(context);
}
} // namespace mlir::tc
