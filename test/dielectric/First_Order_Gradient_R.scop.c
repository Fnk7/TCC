void first_order_gradient_r(int batches,
                            int P,
                            int N,
                            double Gwave[const restrict batches][P][3][N],
                            double DM[const restrict batches][N][N],
                            double wave[const restrict batches][P][N],
                            double GR[const restrict batches][P][3])
{
#pragma scop
if (256 <= batches && batches <= 768 && 150 <= P && P <= 250 && 256 <= N && N <= 768){
  double local_0[batches][P][N];
  for (int c0 = 0; c0 < batches; c0++) {
    for (int c1 = 0; c1 < P; c1++) {
      for (int c3 = 0; c3 < N; c3++) {
        local_0[c0][c1][c3] = 0;
        for (int c2 = 0; c2 < N; c2++) {
          double var0 = DM[c0][c3][c2] + DM[c0][c2][c3];
          double var1 = wave[c0][c1][c2] * var0;
          local_0[c0][c1][c3] = var1 + local_0[c0][c1][c3];
        }
      }
    }
  }
  for (int c0 = 0; c0 < batches; c0++) {
    for (int c1 = 0; c1 < P; c1++) {
      for (int c3 = 0; c3 < 3; c3++) {
        GR[c0][c1][c3] = 0;
        for (int c2 = 0; c2 < N; c2++) {
          double var0 = local_0[c0][c1][c2] * Gwave[c0][c1][c3][c2];
          GR[c0][c1][c3] = var0 + GR[c0][c1][c3];
        }
      }
    }
  }
}
#pragma endscop
}
