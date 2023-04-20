module {
  func.func @first_order_gradient_r_workload_0(%arg0: index, %arg1: index, %arg2: tensor<?x?x3xf64, {size_id = array<i64: 1, 1, 2>}>) -> tensor<?x?x3xf64, {size_id = array<i64: 1, 1, 2>, symmetry = #tc.symm<0~1>}> attributes {num_input_sizes = 2 : i64} {
    %0 = tc.add(%arg2, %arg2) {axesmaps = [(0, 1, 2), (1, 0, 2)], signs = [+, +]} :(tensor<?x?x3xf64, {size_id = array<i64: 1, 1, 2>}>, tensor<?x?x3xf64, {size_id = array<i64: 1, 1, 2>}>) -> tensor<?x?x3xf64, {size_id = array<i64: 1, 1, 2>, symmetry = #tc.symm<0~1>}>
    tc.return %0 : tensor<?x?x3xf64, {size_id = array<i64: 1, 1, 2>, symmetry = #tc.symm<0~1>}>
  }
  func.func @first_order_gradient_r_workload_1(%arg0: index, %arg1: index, %arg2: tensor<?x3x3xf64, {size_id = array<i64: 0, 2, 2>}>) -> tensor<?x3x3xf64, {size_id = array<i64: 0, 2, 2>}> attributes {num_input_sizes = 2 : i64} {
    %0 = tc.permute(%arg2) {axesmap = [(0, 2, 1)]} : tensor<?x3x3xf64, {size_id = array<i64: 0, 2, 2>}> -> tensor<?x3x3xf64, {size_id = array<i64: 0, 2, 2>}>
    tc.return %0 : tensor<?x3x3xf64, {size_id = array<i64: 0, 2, 2>}>
  }
  func.func @first_order_gradient_r(%arg0: index {func.name = "P"}, %arg1: index {func.name = "N"}, %arg2: tensor<?x?x3xf64, {size_id = array<i64: 1, 1, 2>}> {func.name = "DM"}, %arg3: tensor<?x3x?xf64, {size_id = array<i64: 0, 2, 1>}> {func.name = "gradient_basis_wave"}, %arg4: tensor<?x?xf64, {size_id = array<i64: 0, 1>}> {func.name = "wave"}) -> (tensor<?x3x3xf64, {size_id = array<i64: 0, 2, 2>}> {func.name = "R"}) attributes {num_input_indices = 0 : i64, num_input_sizes = 2 : i64, num_input_tensors = 3 : i64, sizes_lower_bound = array<i64: 180, 200, 3>, sizes_upper_bound = array<i64: 200, 300, 3>, wirtebacks = [-1]} {
    %0 = call @first_order_gradient_r_workload_0(%arg0, %arg1, %arg2) : (index, index, tensor<?x?x3xf64, {size_id = array<i64: 1, 1, 2>}>) -> tensor<?x?x3xf64, {size_id = array<i64: 1, 1, 2>, symmetry = #tc.symm<0~1>}>
    %1 = "tblas.batch.gemm"(%arg4, %0) {lhs_trans = false, num_K = 1 : i64, num_M = 1 : i64, num_N = 2 : i64, num_batch = 0 : i64, rhs_trans = false} : (tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, tensor<?x?x3xf64, {size_id = array<i64: 1, 1, 2>, symmetry = #tc.symm<0~1>}>) -> tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 2>}>
    %2 = "tblas.batch.gemm"(%1, %arg3) {lhs_trans = true, num_K = 1 : i64, num_M = 1 : i64, num_N = 1 : i64, num_batch = 1 : i64, rhs_trans = true} : (tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 2>}>, tensor<?x3x?xf64, {size_id = array<i64: 0, 2, 1>}>) -> tensor<?x3x3xf64, {size_id = array<i64: 0, 2, 2>}>
    %3 = call @first_order_gradient_r_workload_1(%arg0, %arg1, %2) : (index, index, tensor<?x3x3xf64, {size_id = array<i64: 0, 2, 2>}>) -> tensor<?x3x3xf64, {size_id = array<i64: 0, 2, 2>}>
    tc.return %3 : tensor<?x3x3xf64, {size_id = array<i64: 0, 2, 2>}>
  }
}
