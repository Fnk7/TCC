#ifndef TCC_DIALECT_UTILS_FUNCUTILS_H
#define TCC_DIALECT_UTILS_FUNCUTILS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::tc::utils {
inline int getNumInputSizes(func::FuncOp funcOp) {
  auto attrDict = funcOp->getAttrDictionary();
  auto attr = attrDict.get("num_input_sizes").cast<IntegerAttr>();
  return attr.getValue().getSExtValue();
}
inline StringRef getFuncOpArgName(func::FuncOp funcOp, int64_t argIdx) {
  auto nameAttr = funcOp.getArgAttr(argIdx, "func.name");
  if (!nameAttr)
    return "";
  return nameAttr.cast<StringAttr>().getValue();
}
inline StringRef getFuncOpResultName(func::FuncOp funcOp, int64_t resultIdx) {
  auto nameAttr = funcOp.getResultAttr(resultIdx, "func.name");
  if (!nameAttr)
    return "";
  return nameAttr.cast<StringAttr>().getValue();
}
}; // namespace mlir::tc::utils

#endif
