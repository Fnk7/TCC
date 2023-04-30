void first_order_h(int arg0,
                   int P,
                   int N,
                   double gradient_rho[const restrict arg0][P][3],
                   double gradient_basis_wave[const restrict arg0][P][3][N],
                   double first_order_gradient_rho[const restrict arg0][P][3],
                   double ptab[const restrict arg0][P],
                   double first_order_potential[const restrict arg0][P],
                   double v2rho2[const restrict arg0][P],
                   double first_order_rho[const restrict arg0][P],
                   double v2rhosigma[const restrict arg0][P],
                   double v2sigma2[const restrict arg0][P],
                   double vsigma[const restrict arg0][P],
                   double wave[const restrict arg0][P][N],
                   double H[const restrict arg0][N][N])
{
#pragma scop
if (777 <= arg0 && arg0 <= 777 && 180 <= P && P <= 200 && 200 <= N && N <= 300){
  double local_0[arg0][P];
  for (int c0 = 0; c0 < arg0; c0++) {
    for (int c1 = 0; c1 < P; c1++) {
      local_0[c0][c1] = 0;
      for (int c2 = 0; c2 < 3; c2++) {
        double var0 = gradient_rho[c0][c1][c2] * first_order_gradient_rho[c0][c1][c2];
        local_0[c0][c1] = var0 + local_0[c0][c1];
      }
    }
  }
  double local_1[arg0][P][N];
  for (int c0 = 0; c0 < arg0; c0++) {
    for (int c1 = 0; c1 < P; c1++) {
      for (int c3 = 0; c3 < N; c3++) {
        local_1[c0][c1][c3] = 0;
        for (int c2 = 0; c2 < 3; c2++) {
          double var0 = v2sigma2[c0][c1] * local_0[c0][c1];
          double var1 = first_order_rho[c0][c1] * v2rhosigma[c0][c1];
          double var2 = var1 + var0;
          double var3 = ptab[c0][c1] * var2;
          double var4 = gradient_rho[c0][c1][c2] * var3;
          double var5 = ptab[c0][c1] * vsigma[c0][c1];
          double var6 = first_order_gradient_rho[c0][c1][c2] * var5;
          double var7 = var6 + var4;
          double var8 = var7 * gradient_basis_wave[c0][c1][c2][c3];
          local_1[c0][c1][c3] = var8 + local_1[c0][c1][c3];
        }
      }
    }
  }
  double local_2[arg0][N][N];
  for (int c0 = 0; c0 < arg0; c0++) {
    for (int c2 = 0; c2 < N; c2++) {
      for (int c3 = 0; c3 < N; c3++) {
        local_2[c0][c2][c3] = 0;
        for (int c1 = 0; c1 < P; c1++) {
          double var0 = v2rhosigma[c0][c1] * local_0[c0][c1];
          double var1 = v2rho2[c0][c1] * first_order_rho[c0][c1];
          double var2 = first_order_potential[c0][c1] + var1;
          double var3 = var2 + var0;
          double var4 = ptab[c0][c1] * var3;
          double var5 = var4 * 0.5;
          double var6 = wave[c0][c1][c3] * var5;
          double var7 = var6 + local_1[c0][c1][c3];
          double var8 = wave[c0][c1][c2] * var7;
          local_2[c0][c2][c3] = var8 + local_2[c0][c2][c3];
        }
      }
    }
  }
  for (int c0 = 0; c0 < arg0; c0++) {
    for (int c1 = 0; c1 < N; c1++) {
      for (int c2 = 0; c2 < N; c2++) {
        H[c0][c1][c2] = local_2[c0][c1][c2] + local_2[c0][c2][c1];
      }
    }
  }
}
#pragma endscop
}
