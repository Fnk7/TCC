#include "TCC/Dialect/ContractionOps.h"

using namespace mlir;
using namespace mlir::contraction;

#include "TCC/Dialect/ContractionOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

void ContractionDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TCC/Dialect/ContractionOps.cpp.inc"
      >();
}

//

static auto unpackIntegerArrayAttr(Attribute attr) {
  auto arrayAttr = attr.cast<ArrayAttr>();
  SmallVector<int64_t> values;
  for (auto &value : arrayAttr) {
    values.emplace_back(value.cast<IntegerAttr>().getValue().getSExtValue());
  }
  return values;
}

static auto getTensorEncodingSizeIDs(Type tensorTy) {
  auto rankedTensorTy = tensorTy.cast<RankedTensorType>();
  auto encodingAttr = rankedTensorTy.getEncoding().cast<ArrayAttr>();
  return unpackIntegerArrayAttr(encodingAttr);
}

using TensorEncodingPair = std::pair<Value, SmallVector<int64_t>>;
static bool compareTensorValueOrder(TensorEncodingPair &p1,
                                    TensorEncodingPair &p2) {
  auto t1 = p1.first;
  auto t2 = p2.first;
  auto &sizeIDs1 = p1.second;
  auto &sizeIDs2 = p2.second;
  for (auto idx :
       llvm::seq<int64_t>(0, std::max(sizeIDs1.size(), sizeIDs2.size()))) {
    if (sizeIDs1[idx] < sizeIDs2[idx])
      return true;
    else if (sizeIDs1[idx] > sizeIDs2[idx])
      return false;
  }
  if (sizeIDs1.size() < sizeIDs2.size()) {
    return true;
  } else if (sizeIDs1.size() > sizeIDs2.size()) {
    return false;
  }
  if (auto b1 = t1.dyn_cast<BlockArgument>()) {
    if (auto b2 = t2.dyn_cast<BlockArgument>()) {
      if (b1.getArgNumber() > b2.getArgNumber()) {
        return false;
      }
    }
    return true;
  } else if (t2.isa<BlockArgument>()) {
    return false;
  }
  return t1.getDefiningOp()->isProperAncestor(t2.getDefiningOp());
}

static auto reorderInputTensors(ArrayRef<Value> tensors) {
  using PairWithOrder = std::pair<TensorEncodingPair, size_t>;
  SmallVector<PairWithOrder> pairs;
  for (auto e : llvm::enumerate(tensors)) {
    auto t = e.value();
    auto p = TensorEncodingPair(t, getTensorEncodingSizeIDs(t.getType()));
    pairs.emplace_back(PairWithOrder(p, e.index()));
  }
  // sort
  std::sort(pairs.begin(), pairs.end(),
            [](PairWithOrder &p1, PairWithOrder &p2) {
              return compareTensorValueOrder(p1.first, p2.first);
            });
  SmallVector<size_t> order;
  for (auto p : pairs) {
    order.emplace_back(p.second);
  }
  return order;
}

static auto getIndexListArrayAttribute(OpBuilder &builder,
                                       ArrayRef<SmallVector<int64_t>> list) {
  SmallVector<Attribute, 4> array;
  for (auto item : list) {
    array.emplace_back(builder.getI64ArrayAttr(item));
  }
  return builder.getArrayAttr(array);
}

static auto permuteTensorType(OpBuilder &builder, TensorType tensorTy,
                              SmallVector<int64_t> &permuteMap) {
  auto shape = tensorTy.getShape();
  auto sizeIDs = getTensorEncodingSizeIDs(tensorTy);
  SmallVector<int64_t> newShape;
  SmallVector<int64_t> newSizeIDs;
  // permuteMap : [3,0,2,1]
  // shape/sizeID [s0,s1,s2,s3] -> [s3,s0,s2,s1]
  for (auto axis : permuteMap) {
    newShape.emplace_back(shape[axis]);
    newSizeIDs.emplace_back(sizeIDs[axis]);
  }
  auto encoding = builder.getI64ArrayAttr(newSizeIDs);
  return RankedTensorType::get(newShape, tensorTy.getElementType(), encoding);
}

struct AddReorderPattern : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> inputs = op.getInputs();
    auto newOrder = reorderInputTensors(inputs);
    bool changed = false;
    for (auto e : llvm::enumerate(newOrder)) {
      if (e.value() != e.index()) {
        changed = true;
        break;
      }
    }
    if (!changed)
      return failure();
    auto signsAttr = op.getSigns();
    auto axesMapsAttr = op.getAxesMaps();
    SmallVector<Value> newInputs;
    SmallVector<Attribute> newSigns;
    SmallVector<Attribute> newAxesMaps;
    for (auto order : newOrder) {
      newInputs.emplace_back(inputs[order]);
      newSigns.emplace_back(signsAttr[order]);
      newAxesMaps.emplace_back(axesMapsAttr[order]);
    }
    auto newSignsAttr = rewriter.getArrayAttr(newSigns);
    auto newAxesMapsAttr = rewriter.getArrayAttr(newAxesMaps);
    rewriter.replaceOpWithNewOp<AddOp>(op, op.getType(), newInputs,
                                       newAxesMapsAttr, newSignsAttr);
    return success();
  }
};

struct AddIndicesCanonicalizePattern : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    auto axesMapsAttr = op.getAxesMaps();
    SmallVector<SmallVector<int64_t>> axesMaps;
    for (auto axesMapAttr : axesMapsAttr) {
      axesMaps.emplace_back(unpackIntegerArrayAttr(axesMapAttr));
    }
    bool isCanonicalized = true;
    int64_t maxAxis = -1;
    // [[1,3,2],[1,0,2]] -> false
    // [[0,1,2],[0,3,2]] -> true
    for (auto &axesMap : axesMaps) {
      bool begin = false;
      for (auto axis : axesMap) {
        if (!begin && maxAxis >= axis) {
          continue;
        }
        begin = true;
        if (++maxAxis != axis) {
          isCanonicalized = false;
          break;
        }
      }
      if (!isCanonicalized) {
        break;
      }
    }
    if (isCanonicalized) {
      return failure();
    }
    // [[1,3,2],[1,0,2]] -> [1,3,2,0]
    SmallVector<int64_t> inverseMap;
    for (auto &axesMap : axesMaps) {
      for (auto axis1 : axesMap) {
        bool found = false;
        for (auto axis2 : inverseMap) {
          if (axis1 == axis2) {
            found = true;
            break;
          }
        }
        if (found)
          continue;
        inverseMap.emplace_back(axis1);
      }
    }
    // [1,3,2,0] -> [3,0,2,1]
    SmallVector<int64_t> permuteMap(inverseMap.size());
    for (auto e : llvm::enumerate(inverseMap)) {
      permuteMap[e.value()] = e.index();
    }
    // [[1,3,2],[1,0,2]] -> [[0,1,2],[0,3,2]]
    for (auto &axesMap : axesMaps) {
      for (auto &axis : axesMap) {
        axis = permuteMap[axis];
      }
    }
    // [s0,s1,s2,s3] -> [s3,s0,s2,s1]
    auto newType = permuteTensorType(rewriter, op.getType(), permuteMap);
    auto newAxesMapsAttr = getIndexListArrayAttribute(rewriter, axesMaps);
    auto newAddOp = rewriter.create<AddOp>(op.getLoc(), newType, op.getInputs(),
                                           newAxesMapsAttr, op.getSigns());
    auto permuteAttr = rewriter.getI64ArrayAttr(permuteMap);
    rewriter.replaceOpWithNewOp<PermuteOp>(op, op.getType(), newAddOp,
                                           permuteAttr);
    return success();
  }
};

void AddOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                        MLIRContext *context) {
  patterns.add<AddReorderPattern, AddIndicesCanonicalizePattern>(context);
}

struct PermutePermuteFoldPattern : public OpRewritePattern<PermuteOp> {
  using OpRewritePattern<PermuteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PermuteOp op,
                                PatternRewriter &rewriter) const override {
    auto input1 = op.getInput();
    if (auto op2 = input1.getDefiningOp<PermuteOp>()) {
      auto &op1 = op;
      auto input2 = op2.getInput();
      auto axesMapAttr1 = op1.getAxesMap();
      auto axesMapAttr2 = op2.getAxesMap();
      auto axesMap1 = unpackIntegerArrayAttr(axesMapAttr1);
      auto axesMap2 = unpackIntegerArrayAttr(axesMapAttr2);
      auto axesMap3 = SmallVector<int64_t>();
      for (auto axis1 : axesMap1) {
        auto axis3 = axesMap2[axis1];
        axesMap3.emplace_back(axis3);
      }
      auto axesMapAttr3 = rewriter.getI64ArrayAttr(axesMap3);
      rewriter.replaceOpWithNewOp<PermuteOp>(op, op.getType(), input2,
                                             axesMapAttr3);
      return success();
    }
    return failure();
  }
};

struct PermuteFoldPattern : public OpRewritePattern<PermuteOp> {
  using OpRewritePattern<PermuteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PermuteOp op,
                                PatternRewriter &rewriter) const override {
    auto axesMapAttr = op.getAxesMap();
    auto axesMap = unpackIntegerArrayAttr(axesMapAttr);
    for (auto e : llvm::enumerate(axesMap)) {
      if (e.index() != (size_t)e.value())
        return failure();
    }
    rewriter.replaceOp(op, op.getInput());
    return success();
  }
};

void PermuteOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                            MLIRContext *context) {
  patterns.add<PermutePermuteFoldPattern, PermuteFoldPattern>(context);
}

//===----------------------------------------------------------------------===//
// Ops
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "TCC/Dialect/ContractionOps.cpp.inc"
