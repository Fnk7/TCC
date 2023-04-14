void first_order_h(int P,
                   int N,
                   double gradient_rho[P][3],
                   double gradient_basis_wave[P][3][N],
                   double first_order_gradient_rho[P][3],
                   double ptab[P],
                   double first_order_potential[P],
                   double v2rho2[P],
                   double first_order_rho[P],
                   double v2rhosigma[P],
                   double v2sigma2[P],
                   double vsigma[P],
                   double wave[P][N],
                   double H[N][N])
{
#pragma scop
  double cst0 = 0;
  double cst1 = 0.5;
  // From Tensor EmptyOp
  double local_0[P];
  // From Linalg FillOp
  for (int c0 = 0; c0 < P; c0++) {
    local_0[c0] = cst0;
  }
  // From Linalg GenericOp
  for (int c0 = 0; c0 < P; c0++) {
    for (int c1 = 0; c1 < 3; c1++) {
      double in0 = gradient_rho[c0][c1];
      double in1 = first_order_gradient_rho[c0][c1];
      double in2 = local_0[c0];
      double var0 = in0 * in1;
      double var1 = var0 + in2;
      local_0[c0] = var1;
    }
  }
  // From Tensor EmptyOp
  double local_1[P][N];
  // From Linalg FillOp
  for (int c0 = 0; c0 < P; c0++) {
    for (int c1 = 0; c1 < N; c1++) {
      local_1[c0][c1] = cst0;
    }
  }
  // From Linalg GenericOp
  for (int c0 = 0; c0 < P; c0++) {
    for (int c2 = 0; c2 < N; c2++) {
      for (int c1 = 0; c1 < 3; c1++) {
        double in0 = first_order_gradient_rho[c0][c1];
        double in1 = ptab[c0];
        double in2 = vsigma[c0];
        double in3 = gradient_rho[c0][c1];
        double in4 = first_order_rho[c0];
        double in5 = v2rhosigma[c0];
        double in6 = v2sigma2[c0];
        double in7 = local_0[c0];
        double in8 = gradient_basis_wave[c0][c1][c2];
        double in9 = local_1[c0][c2];
        double var0 = in6 * in7;
        double var1 = in4 * in5;
        double var2 = var1 + var0;
        double var3 = in1 * var2;
        double var4 = in3 * var3;
        double var5 = in1 * in2;
        double var6 = in0 * var5;
        double var7 = var6 + var4;
        double var8 = var7 * in8;
        double var9 = var8 + in9;
        local_1[c0][c2] = var9;
      }
    }
  }
  // From Tensor EmptyOp
  double local_2[N][N];
  // From Linalg FillOp
  for (int c0 = 0; c0 < N; c0++) {
    for (int c1 = 0; c1 < N; c1++) {
      local_2[c0][c1] = cst0;
    }
  }
  // From Linalg GenericOp
  for (int c1 = 0; c1 < N; c1++) {
    for (int c2 = 0; c2 < N; c2++) {
      for (int c0 = 0; c0 < P; c0++) {
        double in0 = wave[c0][c1];
        double in1 = wave[c0][c2];
        double in2 = ptab[c0];
        double in3 = first_order_potential[c0];
        double in4 = v2rho2[c0];
        double in5 = first_order_rho[c0];
        double in6 = v2rhosigma[c0];
        double in7 = local_0[c0];
        double in8 = local_1[c0][c2];
        double in9 = local_2[c1][c2];
        double var0 = in6 * in7;
        double var1 = in4 * in5;
        double var2 = in3 + var1;
        double var3 = var2 + var0;
        double var4 = in2 * var3;
        double var5 = var4 * cst1;
        double var6 = in1 * var5;
        double var7 = var6 + in8;
        double var8 = in0 * var7;
        double var9 = var8 + in9;
        local_2[c1][c2] = var9;
      }
    }
  }
  // From Tensor EmptyOp
  double local_3[N][N];
  // From Linalg GenericOp
  for (int c0 = 0; c0 < N; c0++) {
    for (int c1 = 0; c1 < N; c1++) {
      double in0 = local_2[c0][c1];
      double in1 = local_2[c1][c0];
      double in2 = local_3[c0][c1];
      double var0 = in0 + in1;
      local_3[c0][c1] = var0;
    }
  }
  // From ReturnOp
  for (int c0 = 0; c0 < N; c0++) {
    for (int c1 = 0; c1 < N; c1++) {
      H[c0][c1] = local_3[c0][c1];
    }
  }
#pragma endscop
}
