#include "TCC/Dialect/TC/IR/TCOps.h"
#include "TCC/Dialect/TC/Transforms/MulOrder.h"
#include "TCC/Dialect/TC/Transforms/Passes.h"
#include "TCC/Dialect/TC/Utils/Builder.h"
#include "TCC/Dialect/TC/Utils/Utils.h"
#include "TCC/Dialect/Utils/FuncUtils.h"

#include "mlir/IR/PatternMatch.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tc-mul-order"

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

struct GroupMulByMulOrdersPattern : public OpRewritePattern<MulOp> {
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
    auto inputs = op.getInputs();
    MulOrders mulOrders = getMulOrdersOfMulOp(op);
    auto &mulTree = mulOrders.getMulTree();
    std::function<MappedInputTy(MulNodeTy)> buildMulTree =
        [&](MulNodeTy node) -> MappedInputTy {
      if (node.second == 0) {
        auto position = node.first;
        LLVM_DEBUG(llvm::dbgs() << "a" << node.first;);
        return {inputs[position], axesMaps[position]};
      }
      MappedInputsTy mappedInputs;
      auto node1 = mulTree[node.first];
      LLVM_DEBUG(llvm::dbgs() << "(";);
      mappedInputs.emplace_back(buildMulTree(node1));
      auto node2 = mulTree[node.second];
      LLVM_DEBUG(llvm::dbgs() << " * ";);
      mappedInputs.emplace_back(buildMulTree(node2));
      LLVM_DEBUG(llvm::dbgs() << ")";);
      return createMulOp(rewriter, mappedInputs);
    };
    auto [result, axesMap] = buildMulTree(mulTree[0]);
    LLVM_DEBUG(llvm::dbgs() << "\n";);
    assert(isOrderedAxesMap(axesMap));
    rewriter.replaceOp(op, result);
    return success();
  }
};

void populateGroupMulByMulOrdersPattern(RewritePatternSet &patterns,
                                        MLIRContext *context) {
  patterns.add<GroupMulByMulOrdersPattern>(context);
}
} // namespace mlir::tc
