void first_order_h(int P,
                   int N,
                   double gradient_basis_wave[P][3][N],
                   double first_order_gradient_rho[P][3][3],
                   double gradient_rho[P][3],
                   double ptab[P],
                   double grid_coord[P][3],
                   double v_hartree_gradient[P][3],
                   double v2rho2[P],
                   double first_order_rho[P][3],
                   double v2rhosigma[P],
                   double v2sigma2[P],
                   double vsigma[P],
                   double wave[P][N],
                   double H[N][N][3])
{
#pragma scop
  double cst0 = 0;
  double cst1 = 0.5;
  // From Tensor EmptyOp
  double local_0[P][3];
  // From Linalg FillOp
  for (int c0 = 0; c0 < P; c0++) {
    for (int c1 = 0; c1 < 3; c1++) {
      local_0[c0][c1] = cst0;
    }
  }
  // From Linalg GenericOp
  for (int c0 = 0; c0 < P; c0++) {
    for (int c2 = 0; c2 < 3; c2++) {
      for (int c1 = 0; c1 < 3; c1++) {
        double in0 = gradient_rho[c0][c1];
        double in1 = first_order_gradient_rho[c0][c1][c2];
        double in2 = local_0[c0][c2];
        double var0 = in0 * in1;
        double var1 = var0 + in2;
        local_0[c0][c2] = var1;
      }
    }
  }
  // From Tensor EmptyOp
  double local_1[P][N][3];
  // From Linalg FillOp
  for (int c0 = 0; c0 < P; c0++) {
    for (int c1 = 0; c1 < N; c1++) {
      for (int c2 = 0; c2 < 3; c2++) {
        local_1[c0][c1][c2] = cst0;
      }
    }
  }
  // From Linalg GenericOp
  for (int c0 = 0; c0 < P; c0++) {
    for (int c2 = 0; c2 < N; c2++) {
      for (int c3 = 0; c3 < 3; c3++) {
        for (int c1 = 0; c1 < 3; c1++) {
          double in0 = gradient_basis_wave[c0][c1][c2];
          double in1 = first_order_gradient_rho[c0][c1][c3];
          double in2 = ptab[c0];
          double in3 = vsigma[c0];
          double in4 = gradient_rho[c0][c1];
          double in5 = local_0[c0][c3];
          double in6 = v2sigma2[c0];
          double in7 = first_order_rho[c0][c3];
          double in8 = v2rhosigma[c0];
          double in9 = local_1[c0][c2][c3];
          double var0 = in2 * in8;
          double var1 = in7 * var0;
          double var2 = in2 * in6;
          double var3 = in5 * var2;
          double var4 = var3 + var1;
          double var5 = in4 * var4;
          double var6 = in2 * in3;
          double var7 = in1 * var6;
          double var8 = var7 + var5;
          double var9 = in0 * var8;
          double var10 = var9 + in9;
          local_1[c0][c2][c3] = var10;
        }
      }
    }
  }
  // From Tensor EmptyOp
  double local_2[N][N][3];
  // From Linalg FillOp
  for (int c0 = 0; c0 < N; c0++) {
    for (int c1 = 0; c1 < N; c1++) {
      for (int c2 = 0; c2 < 3; c2++) {
        local_2[c0][c1][c2] = cst0;
      }
    }
  }
  // From Linalg GenericOp
  for (int c1 = 0; c1 < N; c1++) {
    for (int c2 = 0; c2 < N; c2++) {
      for (int c3 = 0; c3 < 3; c3++) {
        for (int c0 = 0; c0 < P; c0++) {
          double in0 = wave[c0][c1];
          double in1 = ptab[c0];
          double in2 = wave[c0][c2];
          double in3 = grid_coord[c0][c3];
          double in4 = v_hartree_gradient[c0][c3];
          double in5 = v2rho2[c0];
          double in6 = first_order_rho[c0][c3];
          double in7 = v2rhosigma[c0];
          double in8 = local_0[c0][c3];
          double in9 = local_1[c0][c2][c3];
          double in10 = local_2[c1][c2][c3];
          double var0 = in7 * in8;
          double var1 = in5 * in6;
          double var2 = in3 + in4;
          double var3 = var2 + var1;
          double var4 = var3 + var0;
          double var5 = in1 * cst1;
          double var6 = var5 * in2;
          double var7 = var6 * var4;
          double var8 = var7 + in9;
          double var9 = in0 * var8;
          double var10 = var9 + in10;
          local_2[c1][c2][c3] = var10;
        }
      }
    }
  }
  // From Tensor EmptyOp
  double local_3[N][N][3];
  // From Linalg GenericOp
  for (int c0 = 0; c0 < N; c0++) {
    for (int c1 = 0; c1 < N; c1++) {
      for (int c2 = 0; c2 < 3; c2++) {
        double in0 = local_2[c0][c1][c2];
        double in1 = local_2[c1][c0][c2];
        double in2 = local_3[c0][c1][c2];
        double var0 = in0 + in1;
        local_3[c0][c1][c2] = var0;
      }
    }
  }
  // From ReturnOp
  for (int c0 = 0; c0 < N; c0++) {
    for (int c1 = 0; c1 < N; c1++) {
      for (int c2 = 0; c2 < 3; c2++) {
        H[c0][c1][c2] = local_3[c0][c1][c2];
      }
    }
  }
#pragma endscop
}
