void first_order_gradient_r(int batches,
                            int P,
                            int N,
                            double DM[const restrict batches][N][N][3],
                            double gradient_basis_wave[const restrict batches][P][3][N],
                            double wave[const restrict batches][P][N],
                            double R[const restrict batches][P][3][3]) {
#pragma scop
  if (1 && 256 <= batches && batches <= 768 && 150 <= P && P <= 250 && 256 <= N && N <= 768) {
    double local_0[batches][N][N][3];
    {
      for (int c0 = 0; c0 < batches; c0++) {
        for (int c1 = 0; c1 < N; c1++) {
          for (int c2 = 0; c2 < N; c2++) {
            for (int c3 = 0; c3 < 3; c3++) {
              local_0[c0][c1][c2][c3] = DM[c0][c1][c2][c3] + DM[c0][c2][c1][c3];
            }
          }
        }
      }
    }
    // TODO: Emit C for op: tblas.batch.gemm
    // TODO: Emit C for op: tblas.batch.gemm
    {
      for (int c0 = 0; c0 < batches; c0++) {
        for (int c1 = 0; c1 < P; c1++) {
          for (int c2 = 0; c2 < 3; c2++) {
            for (int c3 = 0; c3 < 3; c3++) {
              R[c0][c1][c2][c3] = [c0][c1][c3][c2];
            }
          }
        }
      }
    }
  }
#pragma endscop
}
