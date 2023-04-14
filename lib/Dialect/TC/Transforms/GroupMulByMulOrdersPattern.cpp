#include "TCC/Dialect/TC/IR/TCOps.h"
#include "TCC/Dialect/TC/Transforms/MulOrder.h"
#include "TCC/Dialect/TC/Transforms/Passes.h"
#include "TCC/Dialect/Utils/FuncUtils.h"

#include "mlir/IR/PatternMatch.h"

namespace mlir::tc {
using namespace utils;

static MulOrders getMulOrdersOfMulOp(MulOp mulOp) {
  auto [lbs, ubs] = getSizesRanges(mulOp->getParentOfType<func::FuncOp>());
  auto sizeIDs = getTensorSizeIDs(mulOp.getResult());
  SmallVector<int64_t> sizes;
  for (auto sizeID : sizeIDs)
    sizes.emplace_back(ubs[sizeID]);
  auto axesMaps = mulOp.getAxesMaps();
  return MulOrders(sizes, axesMaps);
}

struct GroupSubsetMulPattern : public OpRewritePattern<MulOp> {
  using OpRewritePattern<MulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {
    auto axesMaps = op.getAxesMaps();
    if (axesMaps.size() <= 2)
      return failure();
    bool isElementwise = true;
    for (auto &axesMap : axesMaps)
      if (axesMap.size() != axesMaps[0].size()) {
        isElementwise = false;
        break;
      }
    if (isElementwise)
      return failure();
    MulOrders mulOrders = getMulOrdersOfMulOp(op);
    return success();
  }
};

void populateGroupSubsetMulPattern(RewritePatternSet &patterns,
                                   MLIRContext *context) {
  patterns.add<GroupSubsetMulPattern>(context);
}
} // namespace mlir::tc
