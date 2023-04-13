void first_order_gradient_r(int P,
                            int N,
                            double DM[N][N][3],
                            double gradient_basis_wave[P][3][N],
                            double wave[P][N],
                            double R[P][3][3])
{
#pragma scop
  double cst0 = 0;
  // From Tensor EmptyOp
  double local_0[P][N][3];
  // From Linalg FillOp
  for (int c0 = 0; c0 < P; c0++) {
    for (int c1 = 0; c1 < N; c1++) {
      for (int c2 = 0; c2 < 3; c2++) {
        local_0[c0][c1][c2] = cst0;
      }
    }
  }
  // From Linalg GenericOp
  for (int c0 = 0; c0 < P; c0++) {
    for (int c2 = 0; c2 < N; c2++) {
      for (int c3 = 0; c3 < 3; c3++) {
        for (int c1 = 0; c1 < N; c1++) {
          double in0 = wave[c0][c1];
          double in1 = DM[c1][c2][c3];
          double in2 = DM[c2][c1][c3];
          double in3 = local_0[c0][c2][c3];
          double var0 = in1 + in2;
          double var1 = in0 * var0;
          double var2 = var1 + in3;
          local_0[c0][c2][c3] = var2;
        }
      }
    }
  }
  // From Tensor EmptyOp
  double local_1[P][3][3];
  // From Linalg FillOp
  for (int c0 = 0; c0 < P; c0++) {
    for (int c1 = 0; c1 < 3; c1++) {
      for (int c2 = 0; c2 < 3; c2++) {
        local_1[c0][c1][c2] = cst0;
      }
    }
  }
  // From Linalg GenericOp
  for (int c0 = 0; c0 < P; c0++) {
    for (int c2 = 0; c2 < 3; c2++) {
      for (int c3 = 0; c3 < 3; c3++) {
        for (int c1 = 0; c1 < N; c1++) {
          double in0 = local_0[c0][c1][c2];
          double in1 = gradient_basis_wave[c0][c3][c1];
          double in2 = local_1[c0][c2][c3];
          double var0 = in0 * in1;
          double var1 = var0 + in2;
          local_1[c0][c2][c3] = var1;
        }
      }
    }
  }
  // From Linalg GenericOp
  for (int c0 = 0; c0 < P; c0++) {
    for (int c1 = 0; c1 < 3; c1++) {
      for (int c2 = 0; c2 < 3; c2++) {
        double in0 = local_1[c0][c2][c1];
        double in1 = local_1[c0][c1][c2];
        local_1[c0][c1][c2] = in0;
      }
    }
  }
  // From ReturnOp
  for (int c0 = 0; c0 < P; c0++) {
    for (int c1 = 0; c1 < 3; c1++) {
      for (int c2 = 0; c2 < 3; c2++) {
        R[c0][c1][c2] = local_1[c0][c1][c2];
      }
    }
  }
#pragma endscop
}
