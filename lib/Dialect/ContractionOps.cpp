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

static auto unpackIntegerArrayAttr(ArrayAttr arrayAttr) {
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

static auto reorderTensors(ArrayRef<Value> tensors) {
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
  for (auto axis : permuteMap) {
    newShape.emplace_back(shape[axis]);
    newSizeIDs.emplace_back(sizeIDs[axis]);
  }
  auto encoding = builder.getI64ArrayAttr(newSizeIDs);
  return RankedTensorType::get(newShape, tensorTy.getElementType(), encoding);
}

struct AddReorder : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> tensors = op.getInputs();
    auto newOrder = reorderTensors(tensors);
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
    SmallVector<Value> newTensors;
    SmallVector<Attribute> newSigns;
    SmallVector<SmallVector<int64_t>> newAxesMaps;
    for (auto order : newOrder) {
      newTensors.emplace_back(tensors[order]);
      newSigns.emplace_back(signsAttr[order]);
      newAxesMaps.emplace_back(
          unpackIntegerArrayAttr(axesMapsAttr[order].cast<ArrayAttr>()));
    }
    auto newSignsAttr = rewriter.getArrayAttr(newSigns);
    auto sizeIDs = getTensorEncodingSizeIDs(op.getType());
    SmallVector<int64_t> inverseMap(sizeIDs.size(), -1);
    for (auto &axesMap : newAxesMaps) {
      for (auto &axis : axesMap) {
        for (auto &ivAxis : inverseMap) {
          if (ivAxis == -1)
            ivAxis = axis;
          if (ivAxis == axis)
            break;
        }
      }
    }
    SmallVector<int64_t> permuteMap(sizeIDs.size());
    for (auto e : llvm::enumerate(inverseMap)) {
      permuteMap[e.value()] = e.index();
    }
    for (auto &axesMap : newAxesMaps) {
      for (auto &axis : axesMap) {
        axis = permuteMap[axis];
      }
    }
    auto newType = permuteTensorType(rewriter, op.getType(), permuteMap);
    auto newAxesMapsAttr = getIndexListArrayAttribute(rewriter, newAxesMaps);
    Value newAddOp = rewriter.create<AddOp>(op.getLoc(), newType, newTensors,
                                            newAxesMapsAttr, newSignsAttr);
    auto permuteAttr = rewriter.getI64ArrayAttr(permuteMap);
    rewriter.replaceOpWithNewOp<PermuteOp>(op, op.getType(), newAddOp,
                                           permuteAttr);
    return success();
  }
};

void AddOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                        MLIRContext *context) {
  patterns.add<AddReorder>(context);
}

struct PermutePermuteOptPattern : public OpRewritePattern<PermuteOp> {
  using OpRewritePattern<PermuteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PermuteOp op,
                                PatternRewriter &rewriter) const override {
    return failure();
  }
};

void PermuteOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                            MLIRContext *context) {
  patterns.add<PermutePermuteOptPattern>(context);
}

//===----------------------------------------------------------------------===//
// Ops
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "TCC/Dialect/ContractionOps.cpp.inc"
