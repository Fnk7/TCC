#include "../common.h"

extern "C" {
void _mlir_ciface_CT(int64_t, int64_t, Tensor2D *, Tensor2D *, Tensor2D *,
                     Tensor2D *);
}

int main() {
  std::srand(0);
  int64_t size0 = 2, size1 = 3;
  auto [_A, A] = createTensorND<2>({size0, size1});
  auto [_B, B] = createTensorND<2>({size0, size1});
  auto [_C, C] = createTensorND<2>({size1, size0});
  auto [_D, D] = createTensorND<2>({size0, size1});
  initTensorND(A);
  initTensorND(B);
  initTensorND(C);
  initTensorND(D);
  std::cout << "A" << std::endl;
  dumpTensorND(A);
  std::cout << "B" << std::endl;
  dumpTensorND(B);
  std::cout << "C" << std::endl;
  dumpTensorND(C);
  _mlir_ciface_CT(size0, size1, &A, &B, &C, &D);
  std::cout << "D" << std::endl;
  dumpTensorND(D);
  double _E[size0][size1];
  for (int i = 0; i < size0; i++) {
    for (int j = 0; j < size1; j++) {
        _E[i][j] = _A.get()[j + i * size1] * _B.get()[j + i * size1];
        _E[i][j] += _A.get()[j + i * size1] * _C.get()[i + j * size0];
    }
  }
  auto E = getTensorND<2>(&_E[0][0], {size0, size1});
  std::cout << "E" << std::endl;
  dumpTensorND(E);
  return !diffTensorND(D, E);
}
