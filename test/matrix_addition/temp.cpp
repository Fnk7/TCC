#include <stddef.h>
#include <stdint.h>

#include <iostream>

template <typename T, size_t N> struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

using Tensor2D = MemRefDescriptor<double, 2>;
using Tensor3D = MemRefDescriptor<double, 3>;
using Tensor2DPtr = Tensor2D *;
using Tensor3DPtr = Tensor3D *;

Tensor2D getTensor2D(double *p, int64_t size0, int64_t size1) {
  Tensor2D tensor;
  tensor.allocated = p;
  tensor.aligned = p;
  tensor.offset = 0;
  tensor.sizes[0] = size0;
  tensor.sizes[1] = size1;
  tensor.strides[0] = size1;
  tensor.strides[1] = 1;
  return tensor;
}

extern "C" {
void _mlir_ciface_CT(int64_t, int64_t, Tensor2D *A, Tensor2D *B, Tensor2D *C);
}

int main() {
  int64_t size0 = 4, size1 = 4;
  double A[size0][size1];
  double B[size0][size1];
  double C[size0][size1];
  auto tA = getTensor2D(&A[0][0], size0, size1);
  auto tB = getTensor2D(&B[0][0], size0, size1);
  auto tC = getTensor2D(&C[0][0], size0, size1);
  for (int i = 0; i < size0; i++) {
    for (int j = 0; j < size1; j++) {
      A[i][j] = i;
      B[i][j] = j;
      C[i][j] = 1;
    }
  }
  std::cout << "A: " << std::endl;
  for (int i = 0; i < size0; i++) {
    for (int j = 0; j < size1; j++) {
      std::cout << A[i][j] << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << "B: " << std::endl;
  for (int i = 0; i < size0; i++) {
    for (int j = 0; j < size1; j++) {
      std::cout << B[i][j] << ", ";
    }
    std::cout << std::endl;
  }
  _mlir_ciface_CT(size0, size1, &tA, &tB, &tC);
  std::cout << "C: " << std::endl;
  for (int i = 0; i < size0; i++) {
    for (int j = 0; j < size1; j++) {
      std::cout << C[i][j] << ", ";
    }
    std::cout << std::endl;
  }
}
