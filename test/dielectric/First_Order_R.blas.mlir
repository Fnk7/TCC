module {
  func.func @first_order_r(%arg0: index {func.name = "batches"}, %arg1: index {func.name = "P"}, %arg2: index {func.name = "N"}, %arg3: tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}> {func.name = "wave"}, %arg4: tensor<?x?x?xf64, {size_id = array<i64: 0, 2, 2>}> {func.name = "DM"}) -> (tensor<?x?xf64, {size_id = array<i64: 0, 1>}> {func.name = "R"}) attributes {num_input_indices = 0 : i64, num_input_sizes = 3 : i64, num_input_tensors = 2 : i64, sizes_lower_bound = array<i64: 256, 150, 256>, sizes_upper_bound = array<i64: 768, 250, 768>, wirtebacks = [-1], workload = 1 : ui64} {
    %0 = "tblas.batch.gemm"(%arg3, %arg4) {lhs_trans = false, num_K = 1 : i64, num_M = 1 : i64, num_N = 1 : i64, num_batch = 1 : i64, rhs_trans = true} : (tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}>, tensor<?x?x?xf64, {size_id = array<i64: 0, 2, 2>}>) -> tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}>
    %1 = "tblas.batch.dot"(%arg3, %0) {num_batch = 2 : i64} : (tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}>, tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}>) -> tensor<?x?xf64, {size_id = array<i64: 0, 1>}>
    tc.return %1 : tensor<?x?xf64, {size_id = array<i64: 0, 1>}>
  }
}
