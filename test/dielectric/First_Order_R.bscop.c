void first_order_r(int batches,
                   int P,
                   int N,
                   double wave[const restrict batches][P][N],
                   double DM[const restrict batches][N][N],
                   double R[const restrict batches][P]) {
#pragma scop
  if (1 && 256 <= batches && batches <= 768 && 150 <= P && P <= 250 && 256 <= N && N <= 768) {
    // TODO: Emit C for op: tblas.batch.gemm
    // TODO: Emit C for op: tblas.batch.dot
  }
#pragma endscop
}
