void first_order_gradient_r_workload_0(int arg0,
                                       int arg1,
                                       double* arg2,
                                       double* ret0)
{
#define arg2(i0, i1) arg2[i0 * arg1 + i1]
#define ret0(i0, i1) ret0[i0 * arg1 + i1]
  for (int i0 = 0; i0 < arg0; i0++) {
    for (int i1 = 0; i1 < arg1; i1++)
      ret0(i0, i1) = arg2(i0, i1) + arg2(i1, i0);
  }
}

void first_order_gradient_r(int arg0,
                            int arg1,
                            double* arg2,
                            double* arg3,
                            double* arg4,
                            double* ret0)
{
  double var0[arg1 * arg1];
  first_order_gradient_r_workload_0(arg0, arg1, arg2, var0);
  double var1[arg0 * arg1];
  {
    int B = 1;
    char transA = 'N';
    char transB = 'T';
    int M = 1 * arg0;
    int N = 1 * arg1;
    int K = 1 * arg1;
    batch_gemm(B, transA, transB, M, N, K, arg4, var0, var1);
  }
  {
    int B = 1 * arg0;
    char trans = 'N';
    int M = 1 * 3;
    int N = 1 * arg1;
    batch_gemv(B, trans, M, N, arg2, var1, ret0);
  }
}

void batch_first_order_gradient_r(int num_batch,
                                  int arg0,
                                  int arg1,
                                  double* arg2,
                                  double* arg3,
                                  double* arg4,
                                  double* ret0)
{
#define arg2(b) (&arg2[arg0 * 3 * arg1 * b])
#define arg3(b) (&arg3[arg1 * arg1 * b])
#define arg4(b) (&arg4[arg0 * arg1 * b])
#define ret0(b) (&ret0[arg0 * 3 * b])
  for (int b = 0; b < num_batch; b++)
    first_order_gradient_r(arg0, arg1, arg2(b), arg3(b), arg4(b), ret0(b));
}
