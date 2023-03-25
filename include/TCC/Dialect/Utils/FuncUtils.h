#ifndef TCC_DIALECT_UTILS_FUNCUTILS_H
#define TCC_DIALECT_UTILS_FUNCUTILS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::tc::utils {
int getNumInputSizes(func::FuncOp funcOp) {
  auto attrDict = funcOp->getAttrDictionary();
  auto attr = attrDict.get("num_input_sizes").cast<IntegerAttr>();
  return attr.getValue().getSExtValue();
}
}; // namespace mlir::tc::utils

#endif
