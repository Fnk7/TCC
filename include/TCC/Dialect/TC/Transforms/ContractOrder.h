#ifndef TCC_DIALECT_TC_TRANSFORMS_CONTRACTORDER_H
#define TCC_DIALECT_TC_TRANSFORMS_CONTRACTORDER_H

#include "TCC/Dialect/TC/Utils/AxesMap.h"

namespace mlir::tc {
using ContractOrder = SmallVector<AxesRefTy>;

class ContractOrders {
public:
  ContractOrders() : impl(nullptr) {}
  ContractOrders(AxesRefTy axes, AxesMapsRefTy axesMaps);
  ContractOrders(const ContractOrders &) = delete;
  ContractOrders &operator=(const ContractOrders &) = delete;
  ContractOrders(ContractOrders &&other)
      : impl(std::exchange(other.impl, nullptr)) {}
  ContractOrders &operator=(ContractOrders &&other) {
    if (this != &other)
      impl = std::exchange(other.impl, nullptr);
    return *this;
  }
  ~ContractOrders();

  int64_t size() { return getNumberOfOrders(); }
  int64_t getNumberOfOrders();
  ContractOrder getOrderByID(int64_t orderID) {
    ContractOrder axesOrder;
    getOrderByID(orderID, axesOrder);
    return axesOrder;
  }
  void getOrderByID(int64_t orderID, ContractOrder &axesOrder);

  void dump();
  void dumpOrders();

private:
  class Impl;
  Impl *impl;
};
} // namespace mlir::tc

#endif
