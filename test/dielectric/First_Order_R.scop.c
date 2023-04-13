void first_order_r(int P,
                   int N,
                   double wave[P][N],
                   double DM[N][N],
                   double R[P])
{
#pragma scop
  double cst0 = 0;
  // From Tensor EmptyOp
  double local_0[P][N];
  // From Linalg FillOp
  for (int c0 = 0; c0 < P; c0++) {
    for (int c1 = 0; c1 < N; c1++) {
      local_0[c0][c1] = cst0;
    }
  }
  // From Linalg GenericOp
  for (int c0 = 0; c0 < P; c0++) {
    for (int c2 = 0; c2 < N; c2++) {
      for (int c1 = 0; c1 < N; c1++) {
        double in0 = wave[c0][c1];
        double in1 = DM[c2][c1];
        double in2 = local_0[c0][c2];
        double var0 = in0 * in1;
        double var1 = var0 + in2;
        local_0[c0][c2] = var1;
      }
    }
  }
  // From Tensor EmptyOp
  double local_1[P];
  // From Linalg FillOp
  for (int c0 = 0; c0 < P; c0++) {
    local_1[c0] = cst0;
  }
  // From Linalg GenericOp
  for (int c0 = 0; c0 < P; c0++) {
    for (int c1 = 0; c1 < N; c1++) {
      double in0 = wave[c0][c1];
      double in1 = local_0[c0][c1];
      double in2 = local_1[c0];
      double var0 = in0 * in1;
      double var1 = var0 + in2;
      local_1[c0] = var1;
    }
  }
  // From ReturnOp
  for (int c0 = 0; c0 < P; c0++) {
    R[c0] = local_1[c0];
  }
#pragma endscop
}
