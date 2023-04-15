#ifndef TCC_DIALECT_TC_UTILS_AXESMAP_H
#define TCC_DIALECT_TC_UTILS_AXESMAP_H

#include "TCC/Dialect/TC/Utils/Tensor.h"
#include "TCC/Dialect/Utils/Axis.h"

namespace mlir::tc::utils {
// Apply axesMap1 on axesMap2, axesMap2 changed
inline void applyAxesMap(AxesMapRefTy axesMap1, AxesMapTy &axesMap2) {
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
// For contractOp axes and related axesmap
inline int64_t getRankFromAxesMaps(AxesMapsTy &axesMaps) {
  int64_t maxAxis = -1;
  for (auto &axesMap : axesMaps) {
    for (auto axis : axesMap) {
      maxAxis = std::max(axis, maxAxis);
    }
  }
  return maxAxis + 1;
}
inline void mapOriginAxis2NewAxis(AxesMapTy &axesMap) {
  int64_t newAxis = 0;
  for (auto &axis : axesMap) {
    if (axis >= 0) {
      axis = newAxis++;
    }
  }
}
inline AxesMapTy getOrigin2NewAxesMap(AxesMapsTy &axesMaps) {
  auto rank = getRankFromAxesMaps(axesMaps);
  AxesMapTy origin2NewMap(rank, -1);
  for (auto &axesMap : axesMaps) {
    for (auto axis : axesMap) {
      origin2NewMap[axis] = axis;
    }
  }
  mapOriginAxis2NewAxis(origin2NewMap);
  return origin2NewMap;
}
inline AxesMapTy getNew2OriginAxesMap(AxesMapTy &origin2NewMap) {
  AxesMapTy new2OriginMap;
  for (auto [originAxis, newAxis] : llvm::enumerate(origin2NewMap)) {
    if (newAxis >= 0) {
      new2OriginMap.emplace_back(originAxis);
    }
  }
  return new2OriginMap;
}
inline auto inverseAxesMap(AxesMapRefTy axesMap) {
  AxesMapTy inverseMap(axesMap.size());
  for (auto e : llvm::enumerate(axesMap)) {
    inverseMap[e.value()] = e.index();
  }
  return inverseMap;
}
inline auto inverseAxesMap(AxesMapRefTy axesMap, int64_t rank) {
  AxesMapTy inverseMap(rank, -1);
  for (auto e : llvm::enumerate(axesMap)) {
    inverseMap[e.value()] = e.index();
  }
  return inverseMap;
}
} // namespace mlir::tc::utils

#endif
