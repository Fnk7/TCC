void first_order_gradient_r(int batches,
                            int P,
                            int N,
                            double Gwave[const restrict batches][P][3][N],
                            double DM[const restrict batches][N][N],
                            double wave[const restrict batches][P][N],
                            double GR[const restrict batches][P][3]) {
#pragma scop
  if (1 && 256 <= batches && batches <= 768 && 150 <= P && P <= 250 && 256 <= N && N <= 768) {
    double local_0[batches][N][N];
    {
      for (int c0 = 0; c0 < batches; c0++) {
        for (int c1 = 0; c1 < N; c1++) {
          for (int c2 = 0; c2 < N; c2++) {
            local_0[c0][c1][c2] = DM[c0][c1][c2] + DM[c0][c2][c1];
          }
        }
      }
    }
    // TODO: Emit C for op: tblas.batch.gemm
    // TODO: Emit C for op: tblas.batch.gemv
  }
#pragma endscop
}
