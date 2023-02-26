#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#define MAX_DUMP_RANK 2
#define MAX_DUMP_SIZE 4
#define MAX_DIFF 0.0001

template <typename T, unsigned N> struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

template <unsigned N> using TensorND = MemRefDescriptor<double, N>;
using Tensor2D = MemRefDescriptor<double, 2>;
using Tensor3D = MemRefDescriptor<double, 3>;
using Tensor4D = MemRefDescriptor<double, 4>;
using Tensor2DPtr = Tensor2D *;
using Tensor3DPtr = Tensor3D *;
using Tensor4DPtr = Tensor4D *;

template <unsigned N>
inline auto getTensorND(double *p, std::vector<int64_t> &sizes) {
  assert(sizes.size() == N);
  using Tensor = TensorND<N>;
  Tensor tensor;
  tensor.allocated = p;
  tensor.aligned = p;
  tensor.offset = 0;
  int64_t stride = 1;
  for (int i = sizes.size() - 1; i >= 0; --i) {
    tensor.sizes[i] = sizes[i];
    tensor.strides[i] = stride;
    stride *= sizes[i];
  }
  return tensor;
}
template <unsigned N>
inline auto getTensorND(double *p, std::vector<int64_t> &&sizes) {
  return getTensorND<N>(p, sizes);
}

template <unsigned N> inline auto createTensorND(std::vector<int64_t> &sizes) {
  size_t total_size = 1;
  for (auto size : sizes) {
    total_size *= size;
  }
  auto p = std::make_unique<double[]>(total_size);
  return std::make_pair(std::move(p), getTensorND<N>(p.get(), sizes));
}
template <unsigned N> inline auto createTensorND(std::vector<int64_t> &&sizes) {
  return createTensorND<N>(sizes);
}

template <unsigned N>
inline void initTensorND(double *p, int64_t *sizes, int64_t *strides) {
  for (int64_t i = 0; i < sizes[0]; ++i) {
    double *p2 = p + strides[0] * i;
    initTensorND<N - 1>(p2, sizes + 1, strides + 1);
  }
}
template <>
inline void initTensorND<0>(double *p, int64_t *sizes, int64_t *strides) {
  p[0] = (std::rand() % 128 - 32) / 128.0;
}
template <unsigned N> inline void initTensorND(TensorND<N> &tensor) {
  initTensorND<N>(tensor.aligned, tensor.sizes, tensor.strides);
}

template <unsigned N>
inline void dumpTensorND(double *p, int64_t *sizes, int64_t *strides) {
  auto max_size = std::min(sizes[0], (int64_t)MAX_DUMP_SIZE);
  for (int64_t i = 0; i < max_size; ++i) {
    if (i == 0 || N <= MAX_DUMP_RANK) {
      double *p2 = p + strides[0] * i;
      dumpTensorND<N - 1>(p2, sizes + 1, strides + 1);
      if (N == 2) {
        std::cout << std::endl;
      }
    }
  }
}
template <>
inline void dumpTensorND<0>(double *p, int64_t *sizes, int64_t *strides) {
  std::cout << std::setw(12) << p[0];
}
template <unsigned N> inline void dumpTensorND(TensorND<N> &tensor) {
  std::cout << std::scientific << std::setprecision(2);
  dumpTensorND<N>(tensor.aligned, tensor.sizes, tensor.strides);
}

template <unsigned N>
inline double asumTensorND(double *p, int64_t *sizes, int64_t *strides) {
  double sum = 0;
  for (int64_t i = 0; i < sizes[0]; ++i) {
    double *p2 = p + strides[0] * i;
    sum += asumTensorND<N - 1>(p2, sizes + 1, strides + 1);
  }
  return sum;
}
template <>
inline double asumTensorND<0>(double *p, int64_t *sizes, int64_t *strides) {
  return std::abs(p[0]);
}
template <unsigned N> inline double asumTensorND(TensorND<N> &tensor) {
  return asumTensorND<N>(tensor.aligned, tensor.sizes, tensor.strides);
}

template <unsigned N>
inline bool diffTensorND(double *p1, double *p2, double limit,
                         std::vector<int64_t> indices, int64_t *sizes,
                         int64_t *strides1, int64_t *strides2, bool print) {
  bool nodiff = true;
  for (int64_t i = 0; i < sizes[0]; ++i) {
    indices[N - 1] = i;
    double *p1_2 = p1 + strides1[0] * i;
    double *p2_2 = p2 + strides2[0] * i;
    nodiff &= diffTensorND<N - 1>(p1_2, p2_2, limit, indices, sizes + 1,
                                  strides1 + 1, strides2 + 1, print);
  }
  return nodiff;
}
template <>
inline bool diffTensorND<0>(double *p1, double *p2, double limit,
                            std::vector<int64_t> indices, int64_t *sizes,
                            int64_t *strides1, int64_t *strides2, bool print) {
  auto v1 = p1[0];
  auto v2 = p2[0];
  auto diff = std::abs(v1 - v2);
  if (diff > limit) {
    if (print) {
      for (int i = indices.size() - 1; i >= 0; --i) {
        std::cout << indices[i];
        std::cout << (i == 0 ? ": " : ", ");
      }
      std::cout << std::setw(14) << v1;
      std::cout << std::setw(14) << v2;
      std::cout << std::endl;
    }
    return false;
  }
  return true;
}
template <unsigned N>
inline bool diffTensorND(TensorND<N> &tensor1, TensorND<N> &tensor2,
                         bool print = true) {
  std::cout << std::scientific << std::setprecision(6);
  double total_size = 1;
  for (int i = 0; i < N; ++i) {
    assert(tensor1.sizes[i] == tensor2.sizes[i] &&
           "Tensors to diff size not equal!");
    total_size *= tensor1.sizes[0];
  }
  auto asum1 = asumTensorND(tensor1);
  auto asum2 = asumTensorND(tensor2);
  auto asum = (asum1 + asum2) / (total_size * 2);
  auto limit = asum * MAX_DIFF;
  std::vector<int64_t> indices(N);
  return diffTensorND<N>(tensor1.aligned, tensor2.aligned, limit, indices,
                         tensor1.sizes, tensor1.strides, tensor2.strides,
                         print);
}
