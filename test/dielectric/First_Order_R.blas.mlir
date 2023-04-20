module {
  func.func @first_order_r(%arg0: index {func.name = "P"}, %arg1: index {func.name = "N"}, %arg2: tensor<?x?xf64, {size_id = array<i64: 0, 1>}> {func.name = "wave"}, %arg3: tensor<?x?xf64, {size_id = array<i64: 1, 1>}> {func.name = "DM"}) -> (tensor<?xf64, {size_id = array<i64: 0>}> {func.name = "R"}) attributes {num_input_indices = 0 : i64, num_input_sizes = 2 : i64, num_input_tensors = 2 : i64, sizes_lower_bound = array<i64: 180, 200>, sizes_upper_bound = array<i64: 200, 300>, wirtebacks = [-1], workload = 1 : ui64} {
    %0 = "tblas.batch.gemm"(%arg2, %arg3) {lhs_trans = false, num_K = 1 : i64, num_M = 1 : i64, num_N = 1 : i64, num_batch = 0 : i64, rhs_trans = true} : (tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, tensor<?x?xf64, {size_id = array<i64: 1, 1>}>) -> tensor<?x?xf64, {size_id = array<i64: 0, 1>}>
    %1 = "tblas.batch.dot"(%arg2, %0) {num_batch = 1 : i64} : (tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, tensor<?x?xf64, {size_id = array<i64: 0, 1>}>) -> tensor<?xf64, {size_id = array<i64: 0>}>
    tc.return %1 : tensor<?xf64, {size_id = array<i64: 0>}>
  }
}
