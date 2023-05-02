void first_order_gradient_r_workload_0(int arg0,
                                       int arg1,
                                       int arg2,
                                       double arg3[const restrict arg0][arg2][arg2],
                                       double ret0[const restrict arg0][arg2][arg2])
{
#pragma scop
if (256 <= arg0 && arg0 <= 768 && 150 <= arg1 && arg1 <= 250 && 256 <= arg2 && arg2 <= 768){
  for (int c0 = 0; c0 < arg0; c0++) {
    for (int c1 = 0; c1 < arg2; c1++) {
      for (int c2 = 0; c2 < arg2; c2++) {
        ret0[c0][c1][c2] = arg3[c0][c1][c2] + arg3[c0][c2][c1];
      }
    }
  }
}
#pragma endscop
}

void first_order_gradient_r(int batches,
                            int P,
                            int N,
                            double Gwave[const restrict batches][P][3][N],
                            double DM[const restrict batches][N][N],
                            double wave[const restrict batches][P][N],
                            double GR[const restrict batches][P][3]) {
  double local_0[batches][N][N];
  first_order_gradient_r_workload_0(batches, P, N, DM, local_0);
  // TODO: Emit C for op: tblas.batch.gemm
  // TODO: Emit C for op: tblas.batch.gemv
}
