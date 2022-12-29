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

#define TC_CANONICALIZE

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

static auto isOrderUnchanged(ArrayRef<Value> inputs,
                             ArrayRef<size_t> newOrder) {
  for (auto e : llvm::enumerate(newOrder)) {
    // bug record : inputs[e.value()] == inputs[e.index()]
    // bug example : [%arg0,%arg0,%arg1], [1,0,2]
    if (e.value() != e.index() && inputs[e.value()] != inputs[e.index()]) {
      return false;
    }
  }
  return true;
}

static auto isOrderedPermuteMap(ArrayRef<int64_t> permuteMap) {
  for (auto e : llvm::enumerate(permuteMap)) {
    if (e.value() != (int64_t)(e.index())) {
      return false;
    }
  }
  return true;
}

static auto inverseAxesMap(ArrayRef<int64_t> axesMap) {
  SmallVector<int64_t> inverseMap(axesMap.size());
  for (auto e : llvm::enumerate(axesMap)) {
    inverseMap[e.value()] = e.index();
  }
  return inverseMap;
}

static auto getIndexListArrayAttribute(OpBuilder &builder,
                                       ArrayRef<SmallVector<int64_t>> list) {
  SmallVector<Attribute, 4> array;
  for (auto item : list) {
    array.emplace_back(builder.getI64ArrayAttr(item));
  }
  return builder.getArrayAttr(array);
}

// PermuteMap : [2,0,1]
// Encoding : [s2,s0,s1] -> [s0,s1,s2]
// PermuteMap : [3,0,2,1]
// Encoding : [s0,s1,s2,s3] -> [s1,s3,s2,s0]
static auto getPermuteInputType(OpBuilder &builder, TensorType resultType,
                                SmallVector<int64_t> &permuteMap) {
  auto shape = resultType.getShape();
  auto sizeIDs = getTensorEncodingSizeIDs(resultType);
  SmallVector<int64_t> newShape(shape.size());
  SmallVector<int64_t> newSizeIDs(sizeIDs.size());
  for (auto [axis, size, sizeID] : llvm::zip(permuteMap, shape, sizeIDs)) {
    newShape[axis] = size;
    newSizeIDs[axis] = sizeID;
  }
  auto encoding = builder.getI64ArrayAttr(newSizeIDs);
  return RankedTensorType::get(newShape, resultType.getElementType(), encoding);
}

struct AddReorderPattern : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> inputs = op.getInputs();
    auto newOrder = reorderInputTensors(inputs);
    if (isOrderUnchanged(inputs, newOrder))
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
    // [s0,s1,s2,s3] -> [s1,s3,s2,s0]
    auto newType = getPermuteInputType(rewriter, op.getType(), permuteMap);
    auto newAxesMapsAttr = getIndexListArrayAttribute(rewriter, axesMaps);
    auto newAddOp = rewriter.create<AddOp>(op.getLoc(), newType, op.getInputs(),
                                           newAxesMapsAttr, op.getSigns());
    auto permuteAttr = rewriter.getI64ArrayAttr(permuteMap);
    rewriter.replaceOpWithNewOp<PermuteOp>(op, op.getType(), newAddOp,
                                           permuteAttr);
    return success();
  }
};

struct AddAddFoldPattern : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    bool change = false;
    SmallVector<Value> newInputs;
    SmallVector<SmallVector<int64_t>> newAxesMaps;
    SmallVector<bool> newSigns;
    for (auto [input, axesMapAttr, signAttr] :
         llvm::zip(op.getInputs(), op.getAxesMaps(), op.getSigns())) {
      auto axesMap = unpackIntegerArrayAttr(axesMapAttr);
      auto sign = signAttr.cast<BoolAttr>().getValue();
      if (auto op2 = input.getDefiningOp<AddOp>()) {
        change = true;
        for (auto [input2, axesMapAttr2, signAttr2] :
             llvm::zip(op2.getInputs(), op2.getAxesMaps(), op2.getSigns())) {
          auto axesMap2 = unpackIntegerArrayAttr(axesMapAttr2);
          auto sign2 = signAttr2.cast<BoolAttr>().getValue();
          newInputs.emplace_back(input2);
          for (auto &axis : axesMap2) {
            axis = axesMap[axis];
          }
          newAxesMaps.emplace_back(axesMap2);
          sign2 = !(sign ^ sign2);
          newSigns.emplace_back(sign2);
        }
      } else {
        newInputs.emplace_back(input);
        newAxesMaps.emplace_back(axesMap);
        newSigns.emplace_back(sign);
      }
    }
    if (!change) {
      return failure();
    }
    auto newAxesMapsAttr = getIndexListArrayAttribute(rewriter, newAxesMaps);
    auto newSignsAttr = rewriter.getBoolArrayAttr(newSigns);
    rewriter.replaceOpWithNewOp<AddOp>(op, op.getType(), newInputs,
                                       newAxesMapsAttr, newSignsAttr);
    return success();
  }
};

struct AddPermuteFoldPattern : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    bool change = false;
    SmallVector<Value> newInputs;
    SmallVector<SmallVector<int64_t>> newAxesMaps;
    for (auto [input, axesMapAttr] :
         llvm::zip(op.getInputs(), op.getAxesMaps())) {
      auto axesMap = unpackIntegerArrayAttr(axesMapAttr);
      if (auto permuteOp = input.getDefiningOp<PermuteOp>()) {
        change = true;
        // [3,0,2,1]
        auto axesMapAttr2 = permuteOp.getAxesMap();
        auto axesMap2 = unpackIntegerArrayAttr(axesMapAttr2);
        auto newAxesMap = SmallVector<int64_t>(axesMap.size());
        // [i0, i1, i2, i3] -> [i1,i3,i2,i0]
        for (auto [axis, axis2] : llvm::zip(axesMap, axesMap2)) {
          newAxesMap[axis2] = axis;
        }
        newInputs.emplace_back(permuteOp.getInput());
        newAxesMaps.emplace_back(newAxesMap);
      } else {
        newInputs.emplace_back(input);
        newAxesMaps.emplace_back(axesMap);
      }
    }
    if (!change) {
      return failure();
    }
    auto newAxesMapsAttr = getIndexListArrayAttribute(rewriter, newAxesMaps);
    rewriter.replaceOpWithNewOp<AddOp>(op, op.getType(), newInputs,
                                       newAxesMapsAttr, op.getSigns());
    return success();
  }
};

void AddOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                        MLIRContext *context) {
#ifdef TC_CANONICALIZE
  patterns.add<AddNegFlodPattern>(context);
  patterns.add<AddPermuteFoldPattern>(context);
  patterns.add<AddAddFoldPattern>(context);
  patterns.add<AddReorderPattern>(context);
  patterns.add<AddIndicesCanonicalizePattern>(context);
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

struct MulReorderPattern : public OpRewritePattern<MulOp> {
  using OpRewritePattern<MulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> inputs = op.getInputs();
    auto newOrder = reorderInputTensors(inputs);
    if (isOrderUnchanged(inputs, newOrder))
      return failure();
    auto axesMapsAttr = op.getAxesMaps();
    SmallVector<Value> newInputs;
    SmallVector<Attribute> newAxesMaps;
    for (auto order : newOrder) {
      newInputs.emplace_back(inputs[order]);
      newAxesMaps.emplace_back(axesMapsAttr[order]);
    }
    auto newAxesMapsAttr = rewriter.getArrayAttr(newAxesMaps);
    rewriter.replaceOpWithNewOp<MulOp>(op, op.getType(), newInputs,
                                       newAxesMapsAttr);
    return success();
  }
};

struct MulIndicesCanonicalizePattern : public OpRewritePattern<MulOp> {
  using OpRewritePattern<MulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {
    auto axesMapsAttr = op.getAxesMaps();
    SmallVector<SmallVector<int64_t>> axesMaps;
    for (auto axesMapAttr : axesMapsAttr) {
      axesMaps.emplace_back(unpackIntegerArrayAttr(axesMapAttr));
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
    auto permuteMap = inverseAxesMap(inverseMap);
    if (isOrderedPermuteMap(permuteMap))
      return failure();
    // [[1,3,2],[1,0,2]] -> [[0,1,2],[0,3,2]]
    for (auto &axesMap : axesMaps) {
      for (auto &axis : axesMap) {
        axis = permuteMap[axis];
      }
    }
    // [s0,s1,s2,s3] -> [s1,s3,s2,s0]
    auto newType = getPermuteInputType(rewriter, op.getType(), permuteMap);
    auto newAxesMapsAttr = getIndexListArrayAttribute(rewriter, axesMaps);
    auto newAddOp = rewriter.create<MulOp>(op.getLoc(), newType, op.getInputs(),
                                           newAxesMapsAttr);
    auto permuteAttr = rewriter.getI64ArrayAttr(permuteMap);
    rewriter.replaceOpWithNewOp<PermuteOp>(op, op.getType(), newAddOp,
                                           permuteAttr);
    return success();
  }
};

void MulOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                        MLIRContext *context) {
#ifdef TC_CANONICALIZE
  patterns.add<MulNegSwapPattern>(context);
  patterns.add<MulReorderPattern>(context);
  patterns.add<MulIndicesCanonicalizePattern>(context);
#endif
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
#ifdef TC_CANONICALIZE
  patterns.add<PermutePermuteFoldPattern, PermuteFoldPattern>(context);
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

void NegOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                        MLIRContext *context) {
#ifdef TC_CANONICALIZE
  patterns.add<NegNegFoldPattern>(context);
#endif
}

static auto inverseAxesMapWithEmpty(ArrayRef<int64_t> axesMap,
                                    int64_t size = -1) {
  if (size < 0)
    size = axesMap.size();
  SmallVector<int64_t> inverseMap(size, -1);
  for (auto e : llvm::enumerate(axesMap)) {
    if (e.value() >= 0) {
      inverseMap[e.value()] = e.index();
    }
  }
  return inverseMap;
}

static void eraseEmptyAxes(SmallVector<int64_t> &axesMap) {
  size_t idx = 0;
  for (auto axis : axesMap) {
    if (axis >= 0) {
      axesMap[idx++] = axis;
    }
  }
  axesMap.truncate(idx);
}

static auto getContractOpResultType(OpBuilder &builder, TensorType inputType,
                                    SmallVector<int64_t> &axes) {
  auto shape = inputType.getShape();
  auto sizeIDs = getTensorEncodingSizeIDs(inputType);
  SmallVector<int64_t> newShape;
  SmallVector<int64_t> newSizeIDs;
  for (auto e : llvm::enumerate(llvm::zip(shape, sizeIDs))) {
    bool found = false;
    for (auto axis : axes) {
      if (axis == (int64_t)(e.index())) {
        found = true;
      }
    }
    if (found)
      continue;
    auto [size, sizeID] = e.value();
    newShape.emplace_back(size);
    newSizeIDs.emplace_back(sizeID);
  }
  auto encoding = builder.getI64ArrayAttr(newSizeIDs);
  return RankedTensorType::get(newShape, inputType.getElementType(), encoding);
}

struct ContractPermuteSwapPattern : public OpRewritePattern<ContractOp> {
  using OpRewritePattern<ContractOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ContractOp op,
                                PatternRewriter &rewriter) const override {
    auto input = op.getInput();
    if (auto permuteOp = input.getDefiningOp<PermuteOp>()) {
      auto input = permuteOp.getInput();
      // [1,2]
      auto axesAttr = op.getAxes();
      auto axes = unpackIntegerArrayAttr(axesAttr);
      // [2,1,3,0,4]
      auto axesMapAttr = permuteOp.getAxesMap();
      auto axesMap = unpackIntegerArrayAttr(axesMapAttr);
      // [1,2] -> [1,3]
      SmallVector<int64_t> newAxes;
      for (auto axis : axes) {
        newAxes.emplace_back(axesMap[axis]);
      }
      // [2,1,3,0,4] -> [2,x,x,0,4]
      // [2,x,x,0,4] -> [3,x,0,x,4]
      // [3,x,0,x,4] -> [3,0,4]
      // [3,0,4] -> [1,x,x,0,2]
      // [1,x,x,0,2] -> [1,0,2]
      for (auto axis : axes) {
        axesMap[axis] = -1;
      }
      // [2,1,3,0,4] -> [3,0,4]
      auto inverseMap = inverseAxesMapWithEmpty(axesMap);
      eraseEmptyAxes(inverseMap);
      // [3,0,4] -> [1,0,2]
      auto newAxesMap = inverseAxesMapWithEmpty(inverseMap, axesMap.size());
      eraseEmptyAxes(newAxesMap);
      auto newType =
          getContractOpResultType(rewriter, input.getType(), newAxes);
      auto newAxesAttr = rewriter.getI64ArrayAttr(newAxes);
      auto contractOp =
          rewriter.create<ContractOp>(op.getLoc(), newType, input, newAxesAttr);
      auto newAxesMapAttr = rewriter.getI64ArrayAttr(newAxesMap);
      rewriter.replaceOpWithNewOp<PermuteOp>(op, op.getType(), contractOp,
                                             newAxesMapAttr);
      return success();
    }
    return failure();
  }
};

struct ContractAddSwapPattern : public OpRewritePattern<ContractOp> {
  using OpRewritePattern<ContractOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ContractOp op,
                                PatternRewriter &rewriter) const override {
    auto input = op.getInput();
    if (auto addOp = input.getDefiningOp<AddOp>()) {
      // sum((j),A(i) + B(j)), not support
      // sum((j),A(i)) + sum((j),B(j)) = size(j) * A(i) + sum((j),B(j))
      auto axesMapsAttr = addOp.getAxesMaps();
      SmallVector<SmallVector<int64_t>> axesMaps;
      for (auto axesMapAttr : axesMapsAttr) {
        axesMaps.emplace_back(unpackIntegerArrayAttr(axesMapAttr));
      }
      size_t numAxis = axesMaps[0].size();
      for (auto &axesMap : axesMaps) {
        if (axesMap.size() != numAxis)
          return failure();
      }
      auto axesAttr = op.getAxes();
      auto axes = unpackIntegerArrayAttr(axesAttr);
      SmallVector<int64_t> oldNewMap;
      for (auto s : llvm::seq<int64_t>(0, numAxis)) {
        oldNewMap.emplace_back(s);
      }
      for (auto axis : axes) {
        oldNewMap[axis] = -1;
      }
      auto newAxis = 0;
      for (auto &axis : oldNewMap) {
        if (axis >= 0) {
          axis = newAxis++;
        }
      }
      SmallVector<Value> newInputs;
      SmallVector<SmallVector<int64_t>> newAxesMaps;
      for (auto [input, axesMap] : llvm::zip(addOp.getInputs(), axesMaps)) {
        SmallVector<int64_t> newAxes;
        SmallVector<int64_t> newAxesMap;
        for (auto e : llvm::enumerate(axesMap)) {
          if (oldNewMap[e.value()] >= 0) {
            newAxesMap.emplace_back(oldNewMap[e.value()]);
          } else {
            newAxes.emplace_back(e.index());
          }
        }
        auto newType = getContractOpResultType(
            rewriter, input.getType().cast<TensorType>(), newAxes);
        auto newAxesAttr = rewriter.getI64ArrayAttr(newAxes);
        auto contractOp = rewriter.create<ContractOp>(op.getLoc(), newType,
                                                      input, newAxesAttr);
        newInputs.emplace_back(contractOp);
        newAxesMaps.emplace_back(newAxesMap);
      }
      auto newAxesMapsAttr = getIndexListArrayAttribute(rewriter, newAxesMaps);
      rewriter.replaceOpWithNewOp<AddOp>(op, op.getType(), newInputs,
                                         newAxesMapsAttr, addOp.getSigns());
      return success();
    }
    return failure();
  }
};

void ContractOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                             MLIRContext *context) {
#ifdef TC_CANONICALIZE
  patterns.add<ContractAddSwapPattern>(context);
  patterns.add<ContractPermuteSwapPattern>(context);
#endif
}

//===----------------------------------------------------------------------===//
// Ops
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "TCC/Dialect/ContractionOps.cpp.inc"
