void first_order_h_workload_0(int arg0,
                              int arg1,
                              int arg2,
                              double arg3[const restrict arg0][arg1][arg2],
                              double arg4[const restrict arg0][arg1],
                              double arg5[const restrict arg0][arg1],
                              double arg6[const restrict arg0][arg1],
                              double arg7[const restrict arg0][arg1],
                              double arg8[const restrict arg0][arg1],
                              double arg9[const restrict arg0][arg1],
                              double arg10[const restrict arg0][arg1],
                              double arg11[const restrict arg0][arg1][3],
                              double arg12[const restrict arg0][arg1][3],
                              double arg13[const restrict arg0][arg1][3][arg2],
                              double ret0[const restrict arg0][arg1][arg2])
{
#pragma scop
if (256 <= arg0 && arg0 <= 768 && 150 <= arg1 && arg1 <= 250 && 256 <= arg2 && arg2 <= 768){
  double local_0[arg0][arg1];
  for (int c0 = 0; c0 < arg0; c0++) {
    for (int c1 = 0; c1 < arg1; c1++) {
      local_0[c0][c1] = 0;
      for (int c2 = 0; c2 < 3; c2++) {
        double var0 = arg11[c0][c1][c2] * arg12[c0][c1][c2];
        local_0[c0][c1] = var0 + local_0[c0][c1];
      }
    }
  }
  double local_1[arg0][arg1][arg2];
  for (int c0 = 0; c0 < arg0; c0++) {
    for (int c1 = 0; c1 < arg1; c1++) {
      for (int c3 = 0; c3 < arg2; c3++) {
        local_1[c0][c1][c3] = 0;
        for (int c2 = 0; c2 < 3; c2++) {
          double var0 = arg10[c0][c1] * local_0[c0][c1];
          double var1 = arg8[c0][c1] * arg9[c0][c1];
          double var2 = var1 + var0;
          double var3 = arg7[c0][c1] * var2;
          double var4 = arg11[c0][c1][c2] * var3;
          double var5 = arg7[c0][c1] * arg6[c0][c1];
          double var6 = arg12[c0][c1][c2] * var5;
          double var7 = var6 + var4;
          double var8 = var7 * arg13[c0][c1][c2][c3];
          local_1[c0][c1][c3] = var8 + local_1[c0][c1][c3];
        }
      }
    }
  }
  for (int c0 = 0; c0 < arg0; c0++) {
    for (int c1 = 0; c1 < arg1; c1++) {
      for (int c2 = 0; c2 < arg2; c2++) {
        double var0 = arg9[c0][c1] * local_0[c0][c1];
        double var1 = arg5[c0][c1] * arg8[c0][c1];
        double var2 = arg4[c0][c1] + var1;
        double var3 = var2 + var0;
        double var4 = arg7[c0][c1] * var3;
        double var5 = var4 * 0.5;
        double var6 = arg3[c0][c1][c2] * var5;
        ret0[c0][c1][c2] = var6 + local_1[c0][c1][c2];
      }
    }
  }
}
#pragma endscop
}

void first_order_h(int batches,
                   int P,
                   int N,
                   double gradient_rho[const restrict batches][P][3],
                   double gradient_basis_wave[const restrict batches][P][3][N],
                   double first_order_gradient_rho[const restrict batches][P][3],
                   double ptab[const restrict batches][P],
                   double first_order_potential[const restrict batches][P],
                   double v2rho2[const restrict batches][P],
                   double first_order_rho[const restrict batches][P],
                   double v2rhosigma[const restrict batches][P],
                   double v2sigma2[const restrict batches][P],
                   double vsigma[const restrict batches][P],
                   double wave[const restrict batches][P][N],
                   double H[const restrict batches][N][N]) {
  double local_0[batches][P][N];
  first_order_h_workload_0(batches, P, N, wave, first_order_potential, v2rho2, vsigma, ptab, first_order_rho, v2rhosigma, v2sigma2, gradient_rho, first_order_gradient_rho, gradient_basis_wave, local_0);
  // TODO: Emit C for op: tblas.batch.syr2k
}
