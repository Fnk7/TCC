#include "TCC/Dialect/TBLAS/IR/TBLASOps.h"
#include "TCC/Dialect/TBLAS/Transforms/Passes.h"
#include "TCC/Dialect/TC/IR/TCOps.h"
#include "TCC/Dialect/TC/Utils/AxesMap.h"
#include "TCC/Dialect/TC/Utils/Utils.h"

namespace mlir::tc {
struct Syr2kPattern : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    auto inputs = op.getInputs();
    auto result = op.getResult();
    if (inputs.size() != 2 || inputs[0] != inputs[1])
      return failure();
    auto input = inputs[0];
    auto batchGemmOp = input.getDefiningOp<BatchGemmOp>();
    if (!batchGemmOp)
      return failure();
    // Square matrix check
    auto numM = batchGemmOp.getNum_M();
    auto numN = batchGemmOp.getNum_N();
    if (numM != numN)
      return failure();
    // Batch check
    auto numBatch = batchGemmOp.getNumBatch();
    auto axesMaps = op.getAxesMaps();
    auto axesMap1 = AxesMapRefTy(axesMaps[0]);
    auto axesMap2 = AxesMapRefTy(axesMaps[1]);
    for (auto axis : llvm::seq<AxisTy>(0, numBatch)) {
      if (axesMap1[axis] != axesMap2[axis])
        return failure();
    }
    // Symmetry check
    auto symmetry = getTensorSymmetry(result);
    if (symmetry.size() < numN)
      return failure();
    SymmetryMapTy symmetryMap;
    for (auto symm : symmetry) {
      symmetryMap[symm.first] = symm.second;
      symmetryMap[symm.second] = symm.first;
    }
    auto inverseMap1 = utils::inverseAxesMap(axesMap1);
    auto inverseMap2 = utils::inverseAxesMap(axesMap2);
    AxisTy offsetB = numBatch;
    AxisTy offsetBN = numBatch + numN;
    AxesMapTy selfMap(numN);
    for (auto axis : llvm::seq<AxisTy>(numBatch, numBatch + numN)) {
      auto axis1 = axesMap1[axis];
      auto axis2 = axesMap2[axis];
      if (!symmetryMap.count(axis1) || symmetryMap[axis1] != axis2)
        return failure();
      auto sAxis1 = axis2;
      auto sAxis2 = axis1;
      auto iAxis1 = inverseMap1[sAxis1];
      auto iAxis2 = inverseMap2[sAxis2];
      if (iAxis1 < offsetBN || iAxis1 != iAxis2)
        return failure();
      selfMap[iAxis1 - offsetBN] = axis - offsetB;
    }
    auto permuteMap = getOrderedAxesMap(axesMap1.size());
    auto numK = batchGemmOp.getNum_K();
    bool lhsTrans = batchGemmOp.getLhsTrans();
    bool rhsTrans = batchGemmOp.getRhsTrans();
    if (!(lhsTrans ^ rhsTrans)) {
      // Contract axes offset
      auto iOffset = rhsTrans ? offsetBN : offsetB;
      auto oOffset = lhsTrans ? offsetB : offsetBN;
      for (auto axis : llvm::seq<AxisTy>(0, numN))
        permuteMap[axis + iOffset] = axis + oOffset;
    }
    AxisTy offsetBK = numBatch + numK;
    {
      // Rhs N axes offset
      auto iOffset = rhsTrans ? offsetB : offsetBK;
      auto oOffset = lhsTrans ? offsetBK : offsetB;
      for (auto [iAxis, oAxis] : llvm::enumerate(selfMap))
        permuteMap[iAxis + iOffset] = oAxis + oOffset;
    }
    auto lhs = batchGemmOp.getLhs();
    auto rhs = batchGemmOp.getRhs();
    if (!isOrderedAxesMap(permuteMap)) {
      auto permuteOp = rewriter.create<PermuteOp>(op.getLoc(), rhs, permuteMap);
      rhs = permuteOp.getResult();
    }
    auto inversePermuteMap = utils::inverseAxesMap(axesMap1);
    auto resultType =
        getResultTypeByAxesMap(rewriter, result, inversePermuteMap);
    auto batchSyr2kOp = rewriter.create<BatchSyr2kOp>(
        op.getLoc(), resultType, lhs, rhs, numBatch, lhsTrans, numN, numK);
    result = batchSyr2kOp.getResult();
    if (!isOrderedAxesMap(axesMap1)) {
      auto permuteMap = axesMaps[0];
      auto permuteOp = rewriter.create<PermuteOp>(op.getLoc(), rhs, permuteMap);
      result = permuteOp.getResult();
    }
    rewriter.replaceOp(op, result);
    rewriter.eraseOp(op);
    return success();
  }
};

void populateSyr2kPattern(RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<Syr2kPattern>(context);
}
} // namespace mlir::tc
