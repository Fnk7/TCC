void first_order_r(int batches,
                   int P,
                   int N,
                   double wave[const restrict batches][P][N],
                   double DM[const restrict batches][N][N],
                   double R[const restrict batches][P]) {
  // TODO: Emit C for op: tblas.batch.gemm
  // TODO: Emit C for op: tblas.batch.dot
}
