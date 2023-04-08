#ifndef TCC_DIALECT_TC_TRANSFORMS_CONTRACTORDER_H
#define TCC_DIALECT_TC_TRANSFORMS_CONTRACTORDER_H

#include "TCC/Dialect/TC/Utils/AxesMap.h"

namespace mlir::tc {
class ContractOrder {
public:
  ContractOrder(AxesRefTy axes, AxesMapsRefTy axesMaps);
  ~ContractOrder();

  void dump();
  int64_t getNumberOfOrders();

private:
  class Impl;
  Impl *impl;
};
} // namespace mlir::tc

#endif
