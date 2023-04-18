#include "TCC/Dialect/TC/IR/TCOps.h"
#include "TCC/Dialect/TC/IR/TCPatterns.h"
#include "TCC/Dialect/TC/Transforms/Passes.h"
#include "TCC/Dialect/TC/Utils/AxesMap.h"
#include "TCC/Dialect/TC/Utils/Utils.h"
#include "TCC/Dialect/Utils/FuncUtils.h"

#include "TCC/Dialect/TBLAS/IR/TBLASOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
// #include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tc-convert-to-tblas-generic"

using namespace mlir;
using namespace mlir::tc;

namespace {

void convertContractOpToTBLASGenericOp(ContractOp ctOp) {
  MulOp mulOp;
  auto input = ctOp.getInput();
  if (!(mulOp = input.getDefiningOp<MulOp>()))
    return;
  auto inputs = mulOp.getInputs();
  if (inputs.size() != 2)
    return;
  auto funcOp = ctOp->getParentOfType<func::FuncOp>();
  auto [lbs, ubs] = utils::getSizesRanges(funcOp);
  auto axes = ctOp.getAxes();
  auto sizeIDs = getTensorSizeIDs(input);
  auto contractSizes = 1;
  for (auto axis : axes)
    contractSizes *= ubs[sizeIDs[axis]];
  if (contractSizes < 8)
    return;
  auto rank = sizeIDs.size();
  auto lhs = inputs[0];
  auto rhs = inputs[1];
  auto axesMaps = mulOp.getAxesMaps();
  auto lhsAxesMap = AxesMapRefTy(axesMaps[0]);
  auto rhsAxesMap = AxesMapRefTy(axesMaps[1]);
  auto lhsInverseAxesMap = utils::inverseAxesMap(lhsAxesMap, rank);
  auto rhsInverseAxesMap = utils::inverseAxesMap(rhsAxesMap, rank);
  AxesTy lhsBatch, rhsBatch;
  AxesTy lhsRest, rhsRest;
  AxesTy lhsContract, rhsContract;
  auto contractAxesMap = getContractOpInput2OutputAxisMap(axes, rank);
  for (auto [iAxis, oAxis] : llvm::enumerate(contractAxesMap)) {
    auto lhsAxis = lhsInverseAxesMap[iAxis];
    auto rhsAxis = rhsInverseAxesMap[iAxis];
    assert(lhsAxis >= 0 || rhsAxis >= 0);
    if (oAxis < 0) {
      assert(lhsAxis >= 0 && rhsAxis >= 0);
      lhsContract.emplace_back(lhsAxis);
      rhsContract.emplace_back(rhsAxis);
    } else if (lhsAxis >= 0 && rhsAxis >= 0) {
      lhsBatch.emplace_back(lhsAxis);
      rhsBatch.emplace_back(rhsAxis);
    } else if (lhsAxis >= 0)
      lhsRest.emplace_back(lhsAxis);
    else if (rhsAxis >= 0)
      rhsRest.emplace_back(rhsAxis);
  }
  AxesMapTy permuteMap;
  for (auto [lhsAxis, rhsAxis] : llvm::zip(lhsBatch, rhsBatch)) {
    auto iAxis1 = lhsAxesMap[lhsAxis];
    auto iAxis2 = rhsAxesMap[rhsAxis];
    assert(iAxis1 == iAxis2);
    auto oAxis = contractAxesMap[iAxis1];
    permuteMap.emplace_back(oAxis);
  }
  for (auto lhsAxis : lhsRest) {
    auto iAxis = lhsAxesMap[lhsAxis];
    auto oAxis = contractAxesMap[iAxis];
    permuteMap.emplace_back(oAxis);
  }
  for (auto rhsAxis : rhsRest) {
    auto iAxis = rhsAxesMap[rhsAxis];
    auto oAxis = contractAxesMap[iAxis];
    permuteMap.emplace_back(oAxis);
  }
  OpBuilder builder(ctOp);
  auto genericOp = builder.create<tblas::GenericOp>(
      ctOp.getLoc(), lhs, rhs, lhsBatch, rhsBatch, lhsRest, rhsRest,
      lhsContract, rhsContract);
  auto result = genericOp.getResult();
  if (!isOrderedAxesMap(permuteMap)) {
    auto permuteOp =
        builder.create<tc::PermuteOp>(ctOp.getLoc(), result, permuteMap);
    result = permuteOp.getResult();
  }
  ctOp.replaceAllUsesWith(result);
}

struct ConvertTCToTBLASPass
    : public PassWrapper<ConvertTCToTBLASPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertTCToTBLASPass)
  void runOnOperation() override {
    SmallVector<tc::ContractOp> worklist;
    for (auto &op : getOperation().front()) {
      if (auto ctOp = dyn_cast_or_null<tc::ContractOp>(&op)) {
        worklist.emplace_back(ctOp);
      }
    }
    for (auto ctOp : worklist)
      convertContractOpToTBLASGenericOp(ctOp);
  }
};

} // namespace

namespace mlir::tc {
std::unique_ptr<Pass> createConvertTCToTBLASPass() {
  return std::make_unique<ConvertTCToTBLASPass>();
}
} // namespace mlir::tc