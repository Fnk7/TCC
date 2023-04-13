#include "TCC/Dialect/TC/Transforms/MulOrder.h"

namespace mlir::tc {
class MulOrders::Impl {
  ArrayRef<int64_t> sizes;
  AxesMapsRefTy axesMaps;
  // axesMaps.size()
  SmallVector<unsigned> axesCodes;
  // 1 << (axesMaps.size())
  SmallVector<double> costs;
  SmallVector<std::pair<unsigned, unsigned>> subPosCodePairs;
  // 1 << (sizes.size())
  SmallVector<double> axesCosts;

public:
  Impl(ArrayRef<int64_t> sizes, AxesMapsRefTy axesMaps)
      : sizes(sizes), axesMaps(axesMaps) {
    auto numPositions = axesMaps.size();
    assert(numPositions <= 8);
    costs.resize((1 << numPositions), -1);
    subPosCodePairs.resize((1 << numPositions));
    for (auto [pos, axesMap] : llvm::enumerate(axesMaps)) {
      auto posCode = encode<PositionTy>(pos);
      axesCodes.emplace_back(encode<AxisTy>(axesMap));
      costs[posCode] = 0;
      subPosCodePairs[posCode] = {0, 0};
    }
    auto numSizes = sizes.size();
    assert(numSizes <= 6);
    axesCosts.resize((1 << numSizes), 0);
  }

  double getCost() {
    unsigned posCode = getMaxCode(axesMaps.size());
    return getCost(posCode);
  }

  double getRestCost(PositionTy position) {
    unsigned restCode = getMaxCode(axesMaps.size());
    unsigned posCode = encode<PositionTy>(position);
    restCode = restCode ^ posCode;
    return getCost(restCode);
  }

  double getCost(unsigned posCode) {
    if (costs[posCode] == -1)
      setCost(posCode);
    return costs[posCode];
  }

  void setCost(unsigned posCode) {
    auto positions = decode<int64_t>(posCode);
    auto axesCode = 0;
    for (auto pos : positions)
      axesCode |= axesCodes[pos];
    auto baseCost = getBaseCost(axesCode);
    // simplify case for size == 2
    if (positions.size() == 2) {
      unsigned posCode1 = encode<PositionTy>(positions[0]);
      unsigned posCode2 = encode<PositionTy>(positions[1]);
      subPosCodePairs[posCode] = {posCode1, posCode2};
      costs[posCode] = baseCost;
      return;
    }
    // set costs[posCode] to max value more
    double minCost = baseCost * (positions.size() - 1);
    SmallVector<unsigned> bitPosCodes;
    for (auto pos : positions)
      bitPosCodes.emplace_back(encode(pos));
    auto maxSubCode = getMaxCode(positions.size() - 1);
    for (unsigned subPosCode = 1; subPosCode <= maxSubCode; subPosCode++) {
      unsigned posCode1 = 0;
      unsigned posCode2 = 0;
      for (auto [subPos, bitPosCode] : llvm::enumerate(bitPosCodes)) {
        auto subBitPosCode = encode(subPos);
        if (subBitPosCode & subPosCode)
          posCode1 |= bitPosCode;
        else
          posCode2 |= bitPosCode;
      }
      assert(posCode1 && posCode2);
      assert((posCode1 | posCode2) == posCode && !(posCode1 & posCode2));
      auto subCost1 = getCost(posCode1);
      auto subCost2 = getCost(posCode2);
      auto subCost = subCost1 + subCost2;
      if (subCost < minCost) {
        minCost = subCost;
        subPosCodePairs[posCode] = {posCode1, posCode2};
      }
    }
    costs[posCode] = baseCost + minCost;
  }

  template <typename T> static unsigned encode(ArrayRef<T> bits) {
    unsigned code = 0;
    for (auto bit : bits)
      code |= (1 << bit);
    return code;
  }
  template <typename T> static unsigned encode(T bit) { return 1 << bit; }
  static unsigned getMaxCode(size_t size) { return (1 << size) - 1; }

  template <typename T> static SmallVector<T> decode(unsigned code) {
    SmallVector<T> result;
    T bit = 0;
    while ((code >> bit)) {
      if ((code >> bit) & 1)
        result.emplace_back(bit);
      bit++;
    }
    return result;
  }

  double getBaseCost(unsigned axesCode) {
    if (axesCosts[axesCode] == 0) {
      auto axes = decode<AxisTy>(axesCode);
      double cost = 1;
      for (auto axis : axes)
        cost *= sizes[axis];
      axesCosts[axesCode] = cost;
    }
    return axesCosts[axesCode];
  }
};

// Constructor and destructor
MulOrders::MulOrders(ArrayRef<int64_t> sizes, AxesMapsRefTy axesMaps)
    : impl(new MulOrders::Impl(sizes, axesMaps)) {}
MulOrders::~MulOrders() { delete impl; }

double MulOrders::getCost() { return impl->getCost(); }
double MulOrders::getCost(PositionsRefTy positions) {
  return impl->getCost(Impl::encode<PositionTy>(positions));
}
double MulOrders::getRestCost(PositionTy position) {
  return impl->getRestCost(position);
}
} // namespace mlir::tc
