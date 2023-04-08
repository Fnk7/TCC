#ifndef TCC_UTILS_SETRELATION_H
#define TCC_UTILS_SETRELATION_H

namespace mlir::tc::utils {
template <typename Set1, typename Set2>
bool isIntersect(Set1 &set1, Set2 &set2) {
  for (auto elem1 : set1)
    for (auto elem2 : set2)
      if (elem1 == elem2)
        return true;
  return false;
}
} // namespace mlir::tc::utils

#endif
