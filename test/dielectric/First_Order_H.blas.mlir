#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
module {
  func.func @first_order_h_workload_0(%arg0: index, %arg1: index, %arg2: index, %arg3: tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}>, %arg4: tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, %arg5: tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, %arg6: tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, %arg7: tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, %arg8: tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, %arg9: tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, %arg10: tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, %arg11: tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 3>}>, %arg12: tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 3>}>, %arg13: tensor<?x?x3x?xf64, {size_id = array<i64: 0, 1, 3, 2>}>) -> tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}> attributes {num_input_sizes = 3 : i64, workload = 2 : ui64} {
    %cst = arith.constant 5.000000e-01 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %0 = tensor.empty(%arg0, %arg1) : tensor<?x?xf64, {size_id = array<i64: 0, 1>}>
    %1 = linalg.fill ins(%cst_0 : f64) outs(%0 : tensor<?x?xf64, {size_id = array<i64: 0, 1>}>) -> tensor<?x?xf64, {size_id = array<i64: 0, 1>}>
    %2 = linalg.generic {indexing_maps = [#map0, #map0, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg11, %arg12 : tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 3>}>, tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 3>}>) outs(%1 : tensor<?x?xf64, {size_id = array<i64: 0, 1>}>) {
    ^bb0(%in: f64, %in_1: f64, %out: f64):
      %7 = arith.mulf %in, %in_1 : f64
      %8 = arith.addf %7, %out : f64
      linalg.yield %8 : f64
    } -> tensor<?x?xf64, {size_id = array<i64: 0, 1>}>
    %3 = tensor.empty(%arg0, %arg1, %arg2) : tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}>
    %4 = linalg.fill ins(%cst_0 : f64) outs(%3 : tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}>) -> tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}>
    %5 = linalg.generic {indexing_maps = [#map2, #map3, #map3, #map2, #map3, #map3, #map3, #map3, #map4, #map5], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%arg12, %arg7, %arg6, %arg11, %arg8, %arg9, %arg10, %2, %arg13 : tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 3>}>, tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 3>}>, tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, tensor<?x?x3x?xf64, {size_id = array<i64: 0, 1, 3, 2>}>) outs(%4 : tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}>) {
    ^bb0(%in: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %in_6: f64, %in_7: f64, %in_8: f64, %out: f64):
      %7 = arith.mulf %in_6, %in_7 : f64
      %8 = arith.mulf %in_4, %in_5 : f64
      %9 = arith.addf %8, %7 : f64
      %10 = arith.mulf %in_1, %9 : f64
      %11 = arith.mulf %in_3, %10 : f64
      %12 = arith.mulf %in_1, %in_2 : f64
      %13 = arith.mulf %in, %12 : f64
      %14 = arith.addf %13, %11 : f64
      %15 = arith.mulf %14, %in_8 : f64
      %16 = arith.addf %15, %out : f64
      linalg.yield %16 : f64
    } -> tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}>
    %6 = linalg.generic {indexing_maps = [#map0, #map1, #map1, #map1, #map1, #map1, #map1, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg3, %arg7, %arg4, %arg5, %arg8, %arg9, %2, %5 : tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}>, tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}>) outs(%3 : tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}>) {
    ^bb0(%in: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %in_6: f64, %in_7: f64, %out: f64):
      %7 = arith.mulf %in_5, %in_6 : f64
      %8 = arith.mulf %in_3, %in_4 : f64
      %9 = arith.addf %in_2, %8 : f64
      %10 = arith.addf %9, %7 : f64
      %11 = arith.mulf %in_1, %10 : f64
      %12 = arith.mulf %11, %cst : f64
      %13 = arith.mulf %in, %12 : f64
      %14 = arith.addf %13, %in_7 : f64
      linalg.yield %14 : f64
    } -> tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}>
    tc.return %6 : tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}>
  }
  func.func @first_order_h(%arg0: index {func.name = "batches"}, %arg1: index {func.name = "P"}, %arg2: index {func.name = "N"}, %arg3: tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 3>}> {func.name = "gradient_rho"}, %arg4: tensor<?x?x3x?xf64, {size_id = array<i64: 0, 1, 3, 2>}> {func.name = "gradient_basis_wave"}, %arg5: tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 3>}> {func.name = "first_order_gradient_rho"}, %arg6: tensor<?x?xf64, {size_id = array<i64: 0, 1>}> {func.name = "ptab"}, %arg7: tensor<?x?xf64, {size_id = array<i64: 0, 1>}> {func.name = "first_order_potential"}, %arg8: tensor<?x?xf64, {size_id = array<i64: 0, 1>}> {func.name = "v2rho2"}, %arg9: tensor<?x?xf64, {size_id = array<i64: 0, 1>}> {func.name = "first_order_rho"}, %arg10: tensor<?x?xf64, {size_id = array<i64: 0, 1>}> {func.name = "v2rhosigma"}, %arg11: tensor<?x?xf64, {size_id = array<i64: 0, 1>}> {func.name = "v2sigma2"}, %arg12: tensor<?x?xf64, {size_id = array<i64: 0, 1>}> {func.name = "vsigma"}, %arg13: tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}> {func.name = "wave"}) -> (tensor<?x?x?xf64, {size_id = array<i64: 0, 2, 2>}> {func.name = "H"}) attributes {num_input_indices = 0 : i64, num_input_sizes = 3 : i64, num_input_tensors = 11 : i64, sizes_lower_bound = array<i64: 256, 150, 256, 3>, sizes_upper_bound = array<i64: 768, 250, 768, 3>, wirtebacks = [-1], workload = 1 : ui64} {
    %0 = call @first_order_h_workload_0(%arg0, %arg1, %arg2, %arg13, %arg7, %arg8, %arg12, %arg6, %arg9, %arg10, %arg11, %arg3, %arg5, %arg4) : (index, index, index, tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}>, tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 3>}>, tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 3>}>, tensor<?x?x3x?xf64, {size_id = array<i64: 0, 1, 3, 2>}>) -> tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}>
    %1 = "tblas.batch.syr2k"(%arg13, %0) {num_K = 1 : i64, num_N = 1 : i64, num_batch = 1 : i64, trans = true} : (tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}>, tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}>) -> tensor<?x?x?xf64, {size_id = array<i64: 0, 2, 2>, symmetry = #tc.symm<1~2>}>
    tc.return %1 : tensor<?x?x?xf64, {size_id = array<i64: 0, 2, 2>, symmetry = #tc.symm<1~2>}>
  }
}
