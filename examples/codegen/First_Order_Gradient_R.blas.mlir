module {
  func.func @first_order_gradient_r_workload_0(%arg0: index, %arg1: index, %arg2: tensor<?x?xf64, {size_id = array<i64: 1, 1>}>) -> tensor<?x?xf64, {size_id = array<i64: 1, 1>, symmetry = #tc.symm<0~1>}> attributes {num_input_sizes = 2 : i64, workload = 2 : ui64} {
    %0 = tc.add(%arg2, %arg2) {axesmaps = [(0, 1), (1, 0)], signs = [+, +]} :(tensor<?x?xf64, {size_id = array<i64: 1, 1>}>, tensor<?x?xf64, {size_id = array<i64: 1, 1>}>) -> tensor<?x?xf64, {size_id = array<i64: 1, 1>, symmetry = #tc.symm<0~1>}>
    tc.return %0 : tensor<?x?xf64, {size_id = array<i64: 1, 1>, symmetry = #tc.symm<0~1>}>
  }
func.func @first_order_gradient_r(
            %arg0: index {func.name = "P"},
            %arg1: index {func.name = "N"},
            %arg2: tensor<?x3x?xf64, [0, 2, 1]> {func.name = "Gwave"},
            %arg3: tensor<?x?xf64, [1, 1]> {func.name = "DM"},
            %arg4: tensor<?x?xf64, [0, 1]> {func.name = "wave"}) 
              -> (tensor<3x?xf64, [2, 0]> {func.name = "GR"})
                attributes {...} {
  %0 = call @first_order_gradient_r_workload_0(%arg0, %arg1, %arg3)
  %1 = "tblas.batch.gemm"(%arg4, %0) attributes {...}
  %2 = "tblas.batch.gemv"(%arg2, %1) attributes {...}
  tc.return %2
}

}
