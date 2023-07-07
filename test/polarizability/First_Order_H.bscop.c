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
#pragma scop
  if (1 && 150 <= P && P <= 250 && 256 <= N && N <= 768) {
    double local_0[P][N][3];
    {
      double local_1[P];
      for (int c0 = 0; c0 < P; c0++) {
        local_1[c0] = ptab[c0] * v2rhosigma[c0];
      }
      double local_2[P][3];
      for (int c0 = 0; c0 < P; c0++) {
        for (int c1 = 0; c1 < 3; c1++) {
          local_2[c0][c1] = first_order_rho[c0][c1] * local_1[c0];
        }
      }
      double local_3[P];
      for (int c0 = 0; c0 < P; c0++) {
        local_3[c0] = ptab[c0] * v2sigma2[c0];
      }
      double local_4[P][3][3];
      for (int c0 = 0; c0 < P; c0++) {
        for (int c1 = 0; c1 < 3; c1++) {
          for (int c2 = 0; c2 < 3; c2++) {
            local_4[c0][c1][c2] = gradient_rho[c0][c1] * first_order_gradient_rho[c0][c1][c2];
          }
        }
      }
      double local_5[P][3];
      for (int c0 = 0; c0 < P; c0++) {
        for (int c2 = 0; c2 < 3; c2++) {
          local_5[c0][c2] = 0;
          for (int c1 = 0; c1 < 3; c1++) {
            local_5[c0][c2] = local_4[c0][c1][c2] + local_5[c0][c2];
          }
        }
      }
      double local_6[P][3];
      for (int c0 = 0; c0 < P; c0++) {
        for (int c1 = 0; c1 < 3; c1++) {
          local_6[c0][c1] = local_5[c0][c1] * local_3[c0];
        }
      }
      double local_7[P][3];
      for (int c0 = 0; c0 < P; c0++) {
        for (int c1 = 0; c1 < 3; c1++) {
          local_7[c0][c1] = local_6[c0][c1] + local_2[c0][c1];
        }
      }
      double local_8[P][3][3];
      for (int c0 = 0; c0 < P; c0++) {
        for (int c1 = 0; c1 < 3; c1++) {
          for (int c2 = 0; c2 < 3; c2++) {
            local_8[c0][c1][c2] = gradient_rho[c0][c1] * local_7[c0][c2];
          }
        }
      }
      double local_9[P];
      for (int c0 = 0; c0 < P; c0++) {
        local_9[c0] = ptab[c0] * vsigma[c0];
      }
      double local_10[P][3][3];
      for (int c0 = 0; c0 < P; c0++) {
        for (int c1 = 0; c1 < 3; c1++) {
          for (int c2 = 0; c2 < 3; c2++) {
            local_10[c0][c1][c2] = first_order_gradient_rho[c0][c1][c2] * local_9[c0];
          }
        }
      }
      double local_11[P][3][3];
      for (int c0 = 0; c0 < P; c0++) {
        for (int c1 = 0; c1 < 3; c1++) {
          for (int c2 = 0; c2 < 3; c2++) {
            local_11[c0][c1][c2] = local_10[c0][c1][c2] + local_8[c0][c1][c2];
          }
        }
      }
      double local_12[P][3][N][3];
      for (int c0 = 0; c0 < P; c0++) {
        for (int c1 = 0; c1 < 3; c1++) {
          for (int c2 = 0; c2 < N; c2++) {
            for (int c3 = 0; c3 < 3; c3++) {
              local_12[c0][c1][c2][c3] = gradient_basis_wave[c0][c1][c2] * local_11[c0][c1][c3];
            }
          }
        }
      }
      double local_13[P][N][3];
      for (int c0 = 0; c0 < P; c0++) {
        for (int c2 = 0; c2 < N; c2++) {
          for (int c3 = 0; c3 < 3; c3++) {
            local_13[c0][c2][c3] = 0;
            for (int c1 = 0; c1 < 3; c1++) {
              local_13[c0][c2][c3] = local_12[c0][c1][c2][c3] + local_13[c0][c2][c3];
            }
          }
        }
      }
      double local_14[P][3];
      for (int c0 = 0; c0 < P; c0++) {
        for (int c1 = 0; c1 < 3; c1++) {
          local_14[c0][c1] = v2rhosigma[c0] * local_5[c0][c1];
        }
      }
      double local_15[P][3];
      for (int c0 = 0; c0 < P; c0++) {
        for (int c1 = 0; c1 < 3; c1++) {
          local_15[c0][c1] = v2rho2[c0] * first_order_rho[c0][c1];
        }
      }
      double local_16[P][3];
      for (int c0 = 0; c0 < P; c0++) {
        for (int c1 = 0; c1 < 3; c1++) {
          double var0 = grid_coord[c0][c1] + v_hartree_gradient[c0][c1];
          double var1 = var0 + local_15[c0][c1];
          local_16[c0][c1] = var1 + local_14[c0][c1];
        }
      }
      double local_17[P];
      for (int c0 = 0; c0 < P; c0++) {
        local_17[c0] =  * ptab[c0];
      }
      double local_18[P][3];
      for (int c0 = 0; c0 < P; c0++) {
        for (int c1 = 0; c1 < 3; c1++) {
          local_18[c0][c1] = local_17[c0] * local_16[c0][c1];
        }
      }
      double local_19[P][N][3];
      for (int c0 = 0; c0 < P; c0++) {
        for (int c1 = 0; c1 < N; c1++) {
          for (int c2 = 0; c2 < 3; c2++) {
            local_19[c0][c1][c2] = wave[c0][c1] * local_18[c0][c2];
          }
        }
      }
      for (int c0 = 0; c0 < P; c0++) {
        for (int c1 = 0; c1 < N; c1++) {
          for (int c2 = 0; c2 < 3; c2++) {
            local_0[c0][c1][c2] = local_19[c0][c1][c2] + local_13[c0][c1][c2];
          }
        }
      }
    }
    // TODO: Emit C for op: tblas.batch.gemm
    {
      for (int c0 = 0; c0 < N; c0++) {
        for (int c1 = 0; c1 < N; c1++) {
          for (int c2 = 0; c2 < 3; c2++) {
            H[c0][c1][c2] = [c0][c1][c2] + [c1][c0][c2];
          }
        }
      }
    }
  }
#pragma endscop
}
