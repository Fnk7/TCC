#ifndef TCC_DIALECT_UTILS_FUNCUTILS_H
#define TCC_DIALECT_UTILS_FUNCUTILS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "llvm/ADT/APSInt.h"

namespace mlir::tc::utils {
inline IntegerAttr getIntegerAttr(MLIRContext *context, int64_t value) {
  return IntegerAttr::get(context, APSInt(APInt(64, value), true));
}
inline int getNumInputSizes(func::FuncOp funcOp) {
  auto attrDict = funcOp->getAttrDictionary();
  auto attr = attrDict.get("num_input_sizes").cast<IntegerAttr>();
  return attr.getValue().getSExtValue();
}
inline void setNumInputSizes(func::FuncOp funcOp, Attribute attr) {
  funcOp->setAttr("num_input_sizes", attr);
}
inline int getWorkloadType(func::FuncOp funcOp) {
  auto attrDict = funcOp->getAttrDictionary();
  if (!attrDict.contains("workload"))
    return 0;
  auto attr = attrDict.get("workload").cast<IntegerAttr>();
  return attr.getValue().getSExtValue();
}
inline void setWorkloadType(func::FuncOp funcOp, int64_t workloadType) {
  auto context = funcOp.getContext();
  funcOp->setAttr("workload", getIntegerAttr(context, workloadType));
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
inline auto getSizesRanges(func::FuncOp funcOp) {
  auto attrDict = funcOp->getAttrDictionary();
  auto lbsAttr = attrDict.get("sizes_lower_bound").cast<DenseI64ArrayAttr>();
  auto ubsAttr = attrDict.get("sizes_upper_bound").cast<DenseI64ArrayAttr>();
  return std::pair(lbsAttr.asArrayRef(), ubsAttr.asArrayRef());
}
inline void setSizesRanges(func::FuncOp funcOp, Attribute lbsAttr,
                           Attribute ubsAttr) {
  funcOp->setAttr("sizes_lower_bound", lbsAttr);
  funcOp->setAttr("sizes_upper_bound", ubsAttr);
}
inline void copyFuncAttributes(func::FuncOp from, func::FuncOp to,
                               ArrayRef<StringRef> attrNames) {
  auto attrDict = from->getAttrDictionary();
  for (auto attrName : attrNames) {
    auto attr = attrDict.get(attrName);
    to->setAttr(attrName, attr);
  }
}
inline auto getTCFuncAttrNames() {
  SmallVector<StringRef> attrNames = {"num_input_sizes"};
  return attrNames;
}
}; // namespace mlir::tc::utils

#endif
