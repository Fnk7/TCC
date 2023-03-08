#include "TCC/Dialect/TC/IR/TCOps.h"
#include "TCC/Dialect/TC/IR/TCPatterns.h"
#include "TCC/Dialect/TC/Utils/Utils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::tc;

template <> struct llvm::DenseMapInfo<I64ArrayTy> {
  static inline I64ArrayTy getEmptyKey() { return I64ArrayTy(1, -2); }
  static inline I64ArrayTy getTombstoneKey() { return I64ArrayTy(1, -2); }
  static unsigned getHashValue(const I64ArrayTy &Val) {
    return llvm::hash_combine_range(Val.begin(), Val.end());
  }
  static bool isEqual(const I64ArrayTy &LHS, const I64ArrayTy &RHS) {
    return LHS == RHS;
  }
};

// apply axesMap1 on axesMap2, axesMap2 changed
static void applyAxesMap(AxesMapTy &axesMap1, AxesMapTy &axesMap2) {
  for (auto &axis : axesMap2) {
    axis = axesMap1[axis];
  }
}
static void applyAxesMap(AxesMapTy &axesMap1, AxesMapsTy &axesMaps) {
  for (auto &axesMap2 : axesMaps) {
    applyAxesMap(axesMap1, axesMap2);
  }
}

using MappedInputTy = std::pair<Value, AxesMapTy>;
static void mapOriginAxis2NewAxis(AxesMapTy &axesMap) {
  int64_t newAxis = 0;
  for (auto &axis : axesMap) {
    if (axis >= 0) {
      axis = newAxis++;
    }
  }
}
static int64_t getRankFromAxesMaps(AxesMapsTy &axesMaps) {
  int64_t maxAxis = -1;
  for (auto &axesMap : axesMaps) {
    for (auto axis : axesMap) {
      maxAxis = std::max(axis, maxAxis);
    }
  }
  return maxAxis + 1;
}
static AxesMapTy getOrigin2NewAxesMap(AxesMapsTy &axesMaps) {
  auto rank = getRankFromAxesMaps(axesMaps);
  AxesMapTy origin2NewMap(rank, -1);
  for (auto &axesMap : axesMaps) {
    for (auto axis : axesMap) {
      origin2NewMap[axis] = axis;
    }
  }
  mapOriginAxis2NewAxis(origin2NewMap);
  return origin2NewMap;
}
static AxesMapTy getNew2OriginAxesMap(AxesMapTy &origin2NewMap) {
  AxesMapTy new2OriginMap;
  for (auto [originAxis, newAxis] : llvm::enumerate(origin2NewMap)) {
    if (newAxis >= 0) {
      new2OriginMap.emplace_back(originAxis);
    }
  }
  return new2OriginMap;
}
static MappedInputTy createMulOp(OpBuilder &builder, ValueRange inputs,
                                 AxesMapsTy &axesMaps) {
  if (inputs.size() == 1) {
    return MappedInputTy(inputs[0], axesMaps[0]);
  }
  AxesMapTy origin2NewMap = getOrigin2NewAxesMap(axesMaps);
  applyAxesMap(origin2NewMap, axesMaps);
  auto mulOp = builder.create<MulOp>(builder.getUnknownLoc(), inputs, axesMaps);
  auto retAxesMap = getNew2OriginAxesMap(origin2NewMap);
  return MappedInputTy(mulOp.getResult(), retAxesMap);
}

struct GroupElementwiseMulPattern : public OpRewritePattern<MulOp> {
  using OpRewritePattern<MulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {
    auto inputs = op.getInputs();
    if (inputs.size() <= 2) {
      return failure();
    }
    auto axesMaps = getAxesMaps(op);
    DenseMap<AxesMapTy, SmallVector<int64_t>> groups;
    for (auto [pos, axesMap] : llvm::enumerate(axesMaps)) {
      auto key = axesMap;
      llvm::sort(key);
      groups[key].emplace_back(pos);
    }
    if (groups.size() == axesMaps.size()) {
      return failure();
    }
    if (groups.size() == 1) {
      return failure();
    }
    SmallVector<Value> newInputs;
    AxesMapsTy newAxesMaps;
    for (auto [_, group] : groups) {
      SmallVector<Value> subInputs;
      AxesMapsTy subAxesMaps;
      for (auto pos : group) {
        subInputs.emplace_back(inputs[pos]);
        subAxesMaps.emplace_back(axesMaps[pos]);
      }
      auto [input, axesMap] = createMulOp(rewriter, subInputs, subAxesMaps);
      newInputs.emplace_back(input);
      newAxesMaps.emplace_back(axesMap);
    }
    rewriter.replaceOpWithNewOp<MulOp>(op, newInputs, newAxesMaps);
    return success();
  }
};

namespace mlir::tc {
void populateGroupElementewiseMulPatterns(RewritePatternSet &patterns,
                                          MLIRContext *context) {
  patterns.add<GroupElementwiseMulPattern>(context);
}
} // namespace mlir::tc
