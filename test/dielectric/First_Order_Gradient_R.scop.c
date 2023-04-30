void first_order_gradient_r(int arg0,
                            int P,
                            int N,
                            double Gwave[const restrict arg0][P][3][N],
                            double DM[const restrict arg0][N][N],
                            double wave[const restrict arg0][P][N],
                            double GR[const restrict arg0][P][3])
{
#pragma scop
if (777 <= arg0 && arg0 <= 777 && 180 <= P && P <= 200 && 200 <= N && N <= 300){
  double local_0[arg0][P][N];
  for (int c0 = 0; c0 < arg0; c0++) {
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
  for (int c0 = 0; c0 < arg0; c0++) {
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
