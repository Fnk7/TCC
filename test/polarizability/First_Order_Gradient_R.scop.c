void first_order_gradient_r(int batches,
                            int P,
                            int N,
                            double DM[const restrict batches][N][N][3],
                            double gradient_basis_wave[const restrict batches][P][3][N],
                            double wave[const restrict batches][P][N],
                            double R[const restrict batches][P][3][3])
{
#pragma scop
if (256 <= batches && batches <= 768 && 150 <= P && P <= 250 && 256 <= N && N <= 768){
  double local_0[batches][P][N][3];
  for (int c0 = 0; c0 < batches; c0++) {
    for (int c1 = 0; c1 < P; c1++) {
      for (int c3 = 0; c3 < N; c3++) {
        for (int c4 = 0; c4 < 3; c4++) {
          local_0[c0][c1][c3][c4] = 0;
          for (int c2 = 0; c2 < N; c2++) {
            double var0 = DM[c0][c2][c3][c4] + DM[c0][c3][c2][c4];
            double var1 = wave[c0][c1][c2] * var0;
            local_0[c0][c1][c3][c4] = var1 + local_0[c0][c1][c3][c4];
          }
        }
      }
    }
  }
  double local_1[batches][P][3][3];
  for (int c0 = 0; c0 < batches; c0++) {
    for (int c1 = 0; c1 < P; c1++) {
      for (int c3 = 0; c3 < 3; c3++) {
        for (int c4 = 0; c4 < 3; c4++) {
          local_1[c0][c1][c3][c4] = 0;
          for (int c2 = 0; c2 < N; c2++) {
            double var0 = local_0[c0][c1][c2][c3] * gradient_basis_wave[c0][c1][c4][c2];
            local_1[c0][c1][c3][c4] = var0 + local_1[c0][c1][c3][c4];
          }
        }
      }
    }
  }
  for (int c0 = 0; c0 < batches; c0++) {
    for (int c1 = 0; c1 < P; c1++) {
      for (int c2 = 0; c2 < 3; c2++) {
        for (int c3 = 0; c3 < 3; c3++) {
          R[c0][c1][c2][c3] = local_1[c0][c1][c3][c2];
        }
      }
    }
  }
}
#pragma endscop
}
