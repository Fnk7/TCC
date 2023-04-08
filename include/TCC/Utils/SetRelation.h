#ifndef TCC_UTILS_SETRELATION_H
#define TCC_UTILS_SETRELATION_H

#include "llvm/ADT/STLExtras.h"

namespace mlir::tc::utils {
template <typename Set1, typename Set2>
bool isIntersect(Set1 &set1, Set2 &set2) {
  for (auto elem : set1)
    if (llvm::find(set2, elem) != set2.end())
      return true;
  return false;
}
template <typename Set1, typename Set2>
bool isSubsetOf(Set1 &set1, Set2 &set2) {
  for (auto elem : set1)
    if (llvm::find(set2, elem) == set2.end())
      return false;
  return true;
}
} // namespace mlir::tc::utils

#endif
