#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
module {
  func.func @first_order_gradient_r_workload_0(%arg0: index, %arg1: index, %arg2: index, %arg3: tensor<?x?x?x3xf64, {size_id = array<i64: 0, 2, 2, 3>}>) -> tensor<?x?x?x3xf64, {size_id = array<i64: 0, 2, 2, 3>, symmetry = #tc.symm<1~2>}> attributes {num_input_sizes = 3 : i64, sizes_lower_bound = array<i64: 256, 150, 256, 3>, sizes_upper_bound = array<i64: 768, 250, 768, 3>, workload = 2 : ui64} {
    %0 = tensor.empty(%arg0, %arg2, %arg2) : tensor<?x?x?x3xf64, {size_id = array<i64: 0, 2, 2, 3>, symmetry = #tc.symm<1~2>}>
    %1 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg3, %arg3 : tensor<?x?x?x3xf64, {size_id = array<i64: 0, 2, 2, 3>}>, tensor<?x?x?x3xf64, {size_id = array<i64: 0, 2, 2, 3>}>) outs(%0 : tensor<?x?x?x3xf64, {size_id = array<i64: 0, 2, 2, 3>, symmetry = #tc.symm<1~2>}>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %2 = arith.addf %in, %in_0 : f64
      linalg.yield %2 : f64
    } -> tensor<?x?x?x3xf64, {size_id = array<i64: 0, 2, 2, 3>, symmetry = #tc.symm<1~2>}>
    tc.return %1 : tensor<?x?x?x3xf64, {size_id = array<i64: 0, 2, 2, 3>, symmetry = #tc.symm<1~2>}>
  }
  func.func @first_order_gradient_r_workload_1(%arg0: index, %arg1: index, %arg2: index, %arg3: tensor<?x?x3x3xf64, {size_id = array<i64: 0, 1, 3, 3>}>) -> tensor<?x?x3x3xf64, {size_id = array<i64: 0, 1, 3, 3>}> attributes {num_input_sizes = 3 : i64, sizes_lower_bound = array<i64: 256, 150, 256, 3>, sizes_upper_bound = array<i64: 768, 250, 768, 3>, workload = 2 : ui64} {
    %0 = tensor.empty(%arg0, %arg1) : tensor<?x?x3x3xf64, {size_id = array<i64: 0, 1, 3, 3>}>
    %1 = linalg.generic {indexing_maps = [#map2, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg3 : tensor<?x?x3x3xf64, {size_id = array<i64: 0, 1, 3, 3>}>) outs(%0 : tensor<?x?x3x3xf64, {size_id = array<i64: 0, 1, 3, 3>}>) {
    ^bb0(%in: f64, %out: f64):
      linalg.yield %in : f64
    } -> tensor<?x?x3x3xf64, {size_id = array<i64: 0, 1, 3, 3>}>
    tc.return %1 : tensor<?x?x3x3xf64, {size_id = array<i64: 0, 1, 3, 3>}>
  }
  func.func @first_order_gradient_r(%arg0: index {func.name = "batches"}, %arg1: index {func.name = "P"}, %arg2: index {func.name = "N"}, %arg3: tensor<?x?x?x3xf64, {size_id = array<i64: 0, 2, 2, 3>}> {func.name = "DM"}, %arg4: tensor<?x?x3x?xf64, {size_id = array<i64: 0, 1, 3, 2>}> {func.name = "gradient_basis_wave"}, %arg5: tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}> {func.name = "wave"}) -> (tensor<?x?x3x3xf64, {size_id = array<i64: 0, 1, 3, 3>}> {func.name = "R"}) attributes {num_input_indices = 0 : i64, num_input_sizes = 3 : i64, num_input_tensors = 3 : i64, sizes_lower_bound = array<i64: 256, 150, 256, 3>, sizes_upper_bound = array<i64: 768, 250, 768, 3>, wirtebacks = [-1], workload = 1 : ui64} {
    %0 = call @first_order_gradient_r_workload_0(%arg0, %arg1, %arg2, %arg3) : (index, index, index, tensor<?x?x?x3xf64, {size_id = array<i64: 0, 2, 2, 3>}>) -> tensor<?x?x?x3xf64, {size_id = array<i64: 0, 2, 2, 3>, symmetry = #tc.symm<1~2>}>
    %1 = "tblas.batch.gemm"(%arg5, %0) {lhs_trans = false, num_K = 1 : i64, num_M = 1 : i64, num_N = 2 : i64, num_batch = 1 : i64, rhs_trans = false} : (tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}>, tensor<?x?x?x3xf64, {size_id = array<i64: 0, 2, 2, 3>, symmetry = #tc.symm<1~2>}>) -> tensor<?x?x?x3xf64, {size_id = array<i64: 0, 1, 2, 3>}>
    %2 = "tblas.batch.gemm"(%1, %arg4) {lhs_trans = true, num_K = 1 : i64, num_M = 1 : i64, num_N = 1 : i64, num_batch = 2 : i64, rhs_trans = true} : (tensor<?x?x?x3xf64, {size_id = array<i64: 0, 1, 2, 3>}>, tensor<?x?x3x?xf64, {size_id = array<i64: 0, 1, 3, 2>}>) -> tensor<?x?x3x3xf64, {size_id = array<i64: 0, 1, 3, 3>}>
    %3 = call @first_order_gradient_r_workload_1(%arg0, %arg1, %arg2, %2) : (index, index, index, tensor<?x?x3x3xf64, {size_id = array<i64: 0, 1, 3, 3>}>) -> tensor<?x?x3x3xf64, {size_id = array<i64: 0, 1, 3, 3>}>
    tc.return %3 : tensor<?x?x3x3xf64, {size_id = array<i64: 0, 1, 3, 3>}>
  }
}
