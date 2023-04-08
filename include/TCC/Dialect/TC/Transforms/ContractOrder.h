#ifndef TCC_DIALECT_TC_TRANSFORMS_CONTRACTORDER_H
#define TCC_DIALECT_TC_TRANSFORMS_CONTRACTORDER_H

#include "TCC/Dialect/TC/Utils/AxesMap.h"

namespace mlir::tc {
class ContractOrders {
public:
  ContractOrders(AxesRefTy axes, AxesMapsRefTy axesMaps);
  ~ContractOrders();

  int64_t getNumberOfOrders();

  void dump();
  void dumpOrders();

private:
  class Impl;
  Impl *impl;
};
} // namespace mlir::tc

#endif
