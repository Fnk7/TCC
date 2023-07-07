void first_order_h_workload_0(int arg0,
                              int arg1,
                              double arg2[const restrict arg0][arg1],
                              double arg3[const restrict arg0][3],
                              double arg4[const restrict arg0][3],
                              double arg5[const restrict arg0],
                              double arg6[const restrict arg0][3][arg1],
                              double arg7[const restrict arg0],
                              double arg8[const restrict arg0][3],
                              double arg9[const restrict arg0][3][3],
                              double arg10[const restrict arg0],
                              double arg11[const restrict arg0][3],
                              double arg12[const restrict arg0],
                              double arg13[const restrict arg0],
                              double ret0[const restrict arg0][arg1][3])
{
#pragma scop
if (150 <= arg0 && arg0 <= 250 && 256 <= arg1 && arg1 <= 768){
  double local_0[arg0];
  for (int c0 = 0; c0 < arg0; c0++) {
    local_0[c0] = arg12[c0] * arg13[c0];
  }
  double local_1[arg0][3];
  for (int c0 = 0; c0 < arg0; c0++) {
    for (int c1 = 0; c1 < 3; c1++) {
      local_1[c0][c1] = arg11[c0][c1] * local_0[c0];
    }
  }
  double local_2[arg0];
  for (int c0 = 0; c0 < arg0; c0++) {
    local_2[c0] = arg12[c0] * arg10[c0];
  }
  double local_3[arg0][3][3];
  for (int c0 = 0; c0 < arg0; c0++) {
    for (int c1 = 0; c1 < 3; c1++) {
      for (int c2 = 0; c2 < 3; c2++) {
        local_3[c0][c1][c2] = arg8[c0][c1] * arg9[c0][c1][c2];
      }
    }
  }
  double local_4[arg0][3];
  for (int c0 = 0; c0 < arg0; c0++) {
    for (int c2 = 0; c2 < 3; c2++) {
      local_4[c0][c2] = 0;
      for (int c1 = 0; c1 < 3; c1++) {
        local_4[c0][c2] = local_3[c0][c1][c2] + local_4[c0][c2];
      }
    }
  }
  double local_5[arg0][3];
  for (int c0 = 0; c0 < arg0; c0++) {
    for (int c1 = 0; c1 < 3; c1++) {
      local_5[c0][c1] = local_4[c0][c1] * local_2[c0];
    }
  }
  double local_6[arg0][3];
  for (int c0 = 0; c0 < arg0; c0++) {
    for (int c1 = 0; c1 < 3; c1++) {
      local_6[c0][c1] = local_5[c0][c1] + local_1[c0][c1];
    }
  }
  double local_7[arg0][3][3];
  for (int c0 = 0; c0 < arg0; c0++) {
    for (int c1 = 0; c1 < 3; c1++) {
      for (int c2 = 0; c2 < 3; c2++) {
        local_7[c0][c1][c2] = arg8[c0][c1] * local_6[c0][c2];
      }
    }
  }
  double local_8[arg0];
  for (int c0 = 0; c0 < arg0; c0++) {
    local_8[c0] = arg12[c0] * arg7[c0];
  }
  double local_9[arg0][3][3];
  for (int c0 = 0; c0 < arg0; c0++) {
    for (int c1 = 0; c1 < 3; c1++) {
      for (int c2 = 0; c2 < 3; c2++) {
        local_9[c0][c1][c2] = arg9[c0][c1][c2] * local_8[c0];
      }
    }
  }
  double local_10[arg0][3][3];
  for (int c0 = 0; c0 < arg0; c0++) {
    for (int c1 = 0; c1 < 3; c1++) {
      for (int c2 = 0; c2 < 3; c2++) {
        local_10[c0][c1][c2] = local_9[c0][c1][c2] + local_7[c0][c1][c2];
      }
    }
  }
  double local_11[arg0][3][arg1][3];
  for (int c0 = 0; c0 < arg0; c0++) {
    for (int c1 = 0; c1 < 3; c1++) {
      for (int c2 = 0; c2 < arg1; c2++) {
        for (int c3 = 0; c3 < 3; c3++) {
          local_11[c0][c1][c2][c3] = arg6[c0][c1][c2] * local_10[c0][c1][c3];
        }
      }
    }
  }
  double local_12[arg0][arg1][3];
  for (int c0 = 0; c0 < arg0; c0++) {
    for (int c2 = 0; c2 < arg1; c2++) {
      for (int c3 = 0; c3 < 3; c3++) {
        local_12[c0][c2][c3] = 0;
        for (int c1 = 0; c1 < 3; c1++) {
          local_12[c0][c2][c3] = local_11[c0][c1][c2][c3] + local_12[c0][c2][c3];
        }
      }
    }
  }
  double local_13[arg0][3];
  for (int c0 = 0; c0 < arg0; c0++) {
    for (int c1 = 0; c1 < 3; c1++) {
      local_13[c0][c1] = arg13[c0] * local_4[c0][c1];
    }
  }
  double local_14[arg0][3];
  for (int c0 = 0; c0 < arg0; c0++) {
    for (int c1 = 0; c1 < 3; c1++) {
      local_14[c0][c1] = arg5[c0] * arg11[c0][c1];
    }
  }
  double local_15[arg0][3];
  for (int c0 = 0; c0 < arg0; c0++) {
    for (int c1 = 0; c1 < 3; c1++) {
      double var0 = arg3[c0][c1] + arg4[c0][c1];
      double var1 = var0 + local_14[c0][c1];
      local_15[c0][c1] = var1 + local_13[c0][c1];
    }
  }
  double local_16[arg0];
  for (int c0 = 0; c0 < arg0; c0++) {
    local_16[c0] =  * arg12[c0];
  }
  double local_17[arg0][3];
  for (int c0 = 0; c0 < arg0; c0++) {
    for (int c1 = 0; c1 < 3; c1++) {
      local_17[c0][c1] = local_16[c0] * local_15[c0][c1];
    }
  }
  double local_18[arg0][arg1][3];
  for (int c0 = 0; c0 < arg0; c0++) {
    for (int c1 = 0; c1 < arg1; c1++) {
      for (int c2 = 0; c2 < 3; c2++) {
        local_18[c0][c1][c2] = arg2[c0][c1] * local_17[c0][c2];
      }
    }
  }
  for (int c0 = 0; c0 < arg0; c0++) {
    for (int c1 = 0; c1 < arg1; c1++) {
      for (int c2 = 0; c2 < 3; c2++) {
        ret0[c0][c1][c2] = local_18[c0][c1][c2] + local_12[c0][c1][c2];
      }
    }
  }
}
#pragma endscop
}

void first_order_h_workload_1(int arg0,
                              int arg1,
                              double arg2[const restrict arg1][arg1][3],
                              double ret0[const restrict arg1][arg1][3])
{
#pragma scop
if (150 <= arg0 && arg0 <= 250 && 256 <= arg1 && arg1 <= 768){
  for (int c0 = 0; c0 < arg1; c0++) {
    for (int c1 = 0; c1 < arg1; c1++) {
      for (int c2 = 0; c2 < 3; c2++) {
        ret0[c0][c1][c2] = arg2[c0][c1][c2] + arg2[c1][c0][c2];
      }
    }
  }
}
#pragma endscop
}

void first_order_h(int P,
                   int N,
                   double gradient_basis_wave[const restrict P][3][N],
                   double first_order_gradient_rho[const restrict P][3][3],
                   double gradient_rho[const restrict P][3],
                   double ptab[const restrict P],
                   double grid_coord[const restrict P][3],
                   double v_hartree_gradient[const restrict P][3],
                   double v2rho2[const restrict P],
                   double first_order_rho[const restrict P][3],
                   double v2rhosigma[const restrict P],
                   double v2sigma2[const restrict P],
                   double vsigma[const restrict P],
                   double wave[const restrict P][N],
                   double H[const restrict N][N][3]) {
  double local_0[P][N][3];
  first_order_h_workload_0(P, N, wave, grid_coord, v_hartree_gradient, v2rho2, gradient_basis_wave, vsigma, gradient_rho, first_order_gradient_rho, v2sigma2, first_order_rho, ptab, v2rhosigma, local_0);
  // TODO: Emit C for op: tblas.batch.gemm
  first_order_h_workload_1(P, N, , H);
}
