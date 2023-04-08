#include "llvm/ADT/SmallVector.h"

namespace {
class UnionFind {
public:
  UnionFind(int n) {
    parent.resize(n);
    for (int i = 0; i < n; ++i) {
      parent[i] = i;
    }
  }

  int find(int x) {
    if (parent[x] != x) {
      parent[x] = find(parent[x]);
    }
    return parent[x];
  }

  void Union(int x, int y) { parent[find(x)] = find(y); }

private:
  llvm::SmallVector<int> parent;
};
} // namespace
