#ifndef TCC_DIALECT_TC_TRANSFORMS_CONTRACTORDER_H
#define TCC_DIALECT_TC_TRANSFORMS_CONTRACTORDER_H

#include <memory>

#include "TCC/Dialect/TC/Utils/AxesMap.h"

namespace mlir::tc {
class ContractOrder {
public:
  ContractOrder(AxesRefTy axes, AxesMapsRefTy axesMaps)
      : impl(std::make_unique<Impl>(axes, axesMaps)) {}

  int64_t getNumberOfOrders();

private:
  class Impl;
  std::unique_ptr<Impl> impl;
};
} // namespace mlir::tc

#endif
