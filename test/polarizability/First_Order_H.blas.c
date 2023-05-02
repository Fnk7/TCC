void first_order_h_workload_0(int arg0,
                              int arg1,
                              int arg2,
                              double arg3[const restrict arg0][arg1][arg2],
                              double arg4[const restrict arg0][arg1][3],
                              double arg5[const restrict arg0][arg1][3],
                              double arg6[const restrict arg0][arg1],
                              double arg7[const restrict arg0][arg1][3][arg2],
                              double arg8[const restrict arg0][arg1],
                              double arg9[const restrict arg0][arg1][3],
                              double arg10[const restrict arg0][arg1][3][3],
                              double arg11[const restrict arg0][arg1],
                              double arg12[const restrict arg0][arg1][3],
                              double arg13[const restrict arg0][arg1],
                              double arg14[const restrict arg0][arg1],
                              double ret0[const restrict arg0][arg1][arg2][3])
{
#pragma scop
if (256 <= arg0 && arg0 <= 768 && 150 <= arg1 && arg1 <= 250 && 256 <= arg2 && arg2 <= 768){
  double local_0[arg0][arg1][3];
  for (int c0 = 0; c0 < arg0; c0++) {
    for (int c1 = 0; c1 < arg1; c1++) {
      for (int c3 = 0; c3 < 3; c3++) {
        local_0[c0][c1][c3] = 0;
        for (int c2 = 0; c2 < 3; c2++) {
          double var0 = arg9[c0][c1][c2] * arg10[c0][c1][c2][c3];
          local_0[c0][c1][c3] = var0 + local_0[c0][c1][c3];
        }
      }
    }
  }
  double local_1[arg0][arg1][arg2][3];
  for (int c0 = 0; c0 < arg0; c0++) {
    for (int c1 = 0; c1 < arg1; c1++) {
      for (int c3 = 0; c3 < arg2; c3++) {
        for (int c4 = 0; c4 < 3; c4++) {
          local_1[c0][c1][c3][c4] = 0;
          for (int c2 = 0; c2 < 3; c2++) {
            double var0 = arg13[c0][c1] * arg14[c0][c1];
            double var1 = arg12[c0][c1][c4] * var0;
            double var2 = arg13[c0][c1] * arg11[c0][c1];
            double var3 = local_0[c0][c1][c4] * var2;
            double var4 = var3 + var1;
            double var5 = arg9[c0][c1][c2] * var4;
            double var6 = arg13[c0][c1] * arg8[c0][c1];
            double var7 = arg10[c0][c1][c2][c4] * var6;
            double var8 = var7 + var5;
            double var9 = arg7[c0][c1][c2][c3] * var8;
            local_1[c0][c1][c3][c4] = var9 + local_1[c0][c1][c3][c4];
          }
        }
      }
    }
  }
  for (int c0 = 0; c0 < arg0; c0++) {
    for (int c1 = 0; c1 < arg1; c1++) {
      for (int c2 = 0; c2 < arg2; c2++) {
        for (int c3 = 0; c3 < 3; c3++) {
          double var0 = arg14[c0][c1] * local_0[c0][c1][c3];
          double var1 = arg6[c0][c1] * arg12[c0][c1][c3];
          double var2 = arg4[c0][c1][c3] + arg5[c0][c1][c3];
          double var3 = var2 + var1;
          double var4 = var3 + var0;
          double var5 = arg13[c0][c1] * 0.5;
          double var6 = var5 * var4;
          double var7 = arg3[c0][c1][c2] * var6;
          ret0[c0][c1][c2][c3] = var7 + local_1[c0][c1][c2][c3];
        }
      }
    }
  }
}
#pragma endscop
}

void first_order_h_workload_1(int arg0,
                              int arg1,
                              int arg2,
                              double arg3[const restrict arg0][arg2][arg2][3],
                              double ret0[const restrict arg0][arg2][arg2][3])
{
#pragma scop
if (256 <= arg0 && arg0 <= 768 && 150 <= arg1 && arg1 <= 250 && 256 <= arg2 && arg2 <= 768){
  for (int c0 = 0; c0 < arg0; c0++) {
    for (int c1 = 0; c1 < arg2; c1++) {
      for (int c2 = 0; c2 < arg2; c2++) {
        for (int c3 = 0; c3 < 3; c3++) {
          ret0[c0][c1][c2][c3] = arg3[c0][c1][c2][c3] + arg3[c0][c2][c1][c3];
        }
      }
    }
  }
}
#pragma endscop
}

void first_order_h(int batches,
                   int P,
                   int N,
                   double gradient_basis_wave[const restrict batches][P][3][N],
                   double first_order_gradient_rho[const restrict batches][P][3][3],
                   double gradient_rho[const restrict batches][P][3],
                   double ptab[const restrict batches][P],
                   double grid_coord[const restrict batches][P][3],
                   double v_hartree_gradient[const restrict batches][P][3],
                   double v2rho2[const restrict batches][P],
                   double first_order_rho[const restrict batches][P][3],
                   double v2rhosigma[const restrict batches][P],
                   double v2sigma2[const restrict batches][P],
                   double vsigma[const restrict batches][P],
                   double wave[const restrict batches][P][N],
                   double H[const restrict batches][N][N][3]) {
  double local_0[batches][P][N][3];
  first_order_h_workload_0(batches, P, N, wave, grid_coord, v_hartree_gradient, v2rho2, gradient_basis_wave, vsigma, gradient_rho, first_order_gradient_rho, v2sigma2, first_order_rho, ptab, v2rhosigma, local_0);
  // TODO: Emit C for op: tblas.batch.gemm
  first_order_h_workload_1(batches, P, N, , H);
}
