#ifndef TCC_DIALECT_TC_TRANSFORMS_MULORDER_H
#define TCC_DIALECT_TC_TRANSFORMS_MULORDER_H

#include "TCC/Dialect/Utils/Axis.h"
#include "TCC/Utils/InputPostion.h"

namespace mlir::tc {
using MulTreeTy = SmallVector<std::pair<PositionTy, PositionTy>>;

class MulOrders {
public:
  MulOrders() : impl(nullptr) {}
  MulOrders(ArrayRef<int64_t> sizes, AxesMapsRefTy axesMaps);
  MulOrders(const MulOrders &) = delete;
  MulOrders &operator=(const MulOrders &) = delete;
  MulOrders(MulOrders &&other) : impl(std::exchange(other.impl, nullptr)) {}
  MulOrders &operator=(MulOrders &&other) {
    if (this != &other)
      impl = std::exchange(other.impl, nullptr);
    return *this;
  }
  ~MulOrders();

  double getCost();
  double getCost(PositionsRefTy positions);
  double getRestCost(PositionTy position);
  MulTreeTy &getMulTree();

private:
  class Impl;
  Impl *impl;
};
} // namespace mlir::tc

#endif
