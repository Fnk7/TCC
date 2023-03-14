#ifndef TCC_DIALECT_TC_UTILS_AXESMAP_H
#define TCC_DIALECT_TC_UTILS_AXESMAP_H

namespace mlir::tc::utils {
// apply axesMap1 on axesMap2, axesMap2 changed
inline void applyAxesMap(AxesMapTy &axesMap1, AxesMapTy &axesMap2) {
  for (auto &axis : axesMap2) {
    axis = axesMap1[axis];
  }
}
inline void applyAxesMapWithEmpty(AxesMapTy &axesMap1, AxesMapTy &axesMap2) {
  for (auto &axis : axesMap2) {
    if (axis >= 0) {
      axis = axesMap1[axis];
    }
  }
}
inline void applyAxesMap(AxesMapTy &axesMap1, AxesMapsTy &axesMaps) {
  for (auto &axesMap2 : axesMaps) {
    applyAxesMap(axesMap1, axesMap2);
  }
}
} // namespace mlir::tc::utils

#endif
