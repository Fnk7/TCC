#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>
#map6 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
#map7 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>
#map8 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
#map10 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
module {
  func.func @first_order_h_workload_0(%arg0: index, %arg1: index, %arg2: index, %arg3: tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}>, %arg4: tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 3>}>, %arg5: tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 3>}>, %arg6: tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, %arg7: tensor<?x?x3x?xf64, {size_id = array<i64: 0, 1, 3, 2>}>, %arg8: tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, %arg9: tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 3>}>, %arg10: tensor<?x?x3x3xf64, {size_id = array<i64: 0, 1, 3, 3>}>, %arg11: tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, %arg12: tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 3>}>, %arg13: tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, %arg14: tensor<?x?xf64, {size_id = array<i64: 0, 1>}>) -> tensor<?x?x?x3xf64, {size_id = array<i64: 0, 1, 2, 3>}> attributes {num_input_sizes = 3 : i64, workload = 2 : ui64} {
    %cst = arith.constant 5.000000e-01 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %0 = tensor.empty(%arg0, %arg1) : tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 3>}>
    %1 = linalg.fill ins(%cst_0 : f64) outs(%0 : tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 3>}>) -> tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 3>}>
    %2 = linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%arg9, %arg10 : tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 3>}>, tensor<?x?x3x3xf64, {size_id = array<i64: 0, 1, 3, 3>}>) outs(%1 : tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 3>}>) {
    ^bb0(%in: f64, %in_1: f64, %out: f64):
      %7 = arith.mulf %in, %in_1 : f64
      %8 = arith.addf %7, %out : f64
      linalg.yield %8 : f64
    } -> tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 3>}>
    %3 = tensor.empty(%arg0, %arg1, %arg2) : tensor<?x?x?x3xf64, {size_id = array<i64: 0, 1, 2, 3>}>
    %4 = linalg.fill ins(%cst_0 : f64) outs(%3 : tensor<?x?x?x3xf64, {size_id = array<i64: 0, 1, 2, 3>}>) -> tensor<?x?x?x3xf64, {size_id = array<i64: 0, 1, 2, 3>}>
    %5 = linalg.generic {indexing_maps = [#map3, #map4, #map5, #map5, #map6, #map7, #map5, #map7, #map5, #map8], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel"]} ins(%arg7, %arg10, %arg13, %arg8, %arg9, %2, %arg11, %arg12, %arg14 : tensor<?x?x3x?xf64, {size_id = array<i64: 0, 1, 3, 2>}>, tensor<?x?x3x3xf64, {size_id = array<i64: 0, 1, 3, 3>}>, tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 3>}>, tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 3>}>, tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 3>}>, tensor<?x?xf64, {size_id = array<i64: 0, 1>}>) outs(%4 : tensor<?x?x?x3xf64, {size_id = array<i64: 0, 1, 2, 3>}>) {
    ^bb0(%in: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %in_6: f64, %in_7: f64, %in_8: f64, %out: f64):
      %7 = arith.mulf %in_2, %in_8 : f64
      %8 = arith.mulf %in_7, %7 : f64
      %9 = arith.mulf %in_2, %in_6 : f64
      %10 = arith.mulf %in_5, %9 : f64
      %11 = arith.addf %10, %8 : f64
      %12 = arith.mulf %in_4, %11 : f64
      %13 = arith.mulf %in_2, %in_3 : f64
      %14 = arith.mulf %in_1, %13 : f64
      %15 = arith.addf %14, %12 : f64
      %16 = arith.mulf %in, %15 : f64
      %17 = arith.addf %16, %out : f64
      linalg.yield %17 : f64
    } -> tensor<?x?x?x3xf64, {size_id = array<i64: 0, 1, 2, 3>}>
    %6 = linalg.generic {indexing_maps = [#map0, #map9, #map2, #map2, #map9, #map2, #map9, #map2, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg3, %arg13, %arg4, %arg5, %arg6, %arg12, %arg14, %2, %5 : tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}>, tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 3>}>, tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 3>}>, tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 3>}>, tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 3>}>, tensor<?x?x?x3xf64, {size_id = array<i64: 0, 1, 2, 3>}>) outs(%3 : tensor<?x?x?x3xf64, {size_id = array<i64: 0, 1, 2, 3>}>) {
    ^bb0(%in: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %in_6: f64, %in_7: f64, %in_8: f64, %out: f64):
      %7 = arith.mulf %in_6, %in_7 : f64
      %8 = arith.mulf %in_4, %in_5 : f64
      %9 = arith.addf %in_2, %in_3 : f64
      %10 = arith.addf %9, %8 : f64
      %11 = arith.addf %10, %7 : f64
      %12 = arith.mulf %in_1, %cst : f64
      %13 = arith.mulf %12, %11 : f64
      %14 = arith.mulf %in, %13 : f64
      %15 = arith.addf %14, %in_8 : f64
      linalg.yield %15 : f64
    } -> tensor<?x?x?x3xf64, {size_id = array<i64: 0, 1, 2, 3>}>
    tc.return %6 : tensor<?x?x?x3xf64, {size_id = array<i64: 0, 1, 2, 3>}>
  }
  func.func @first_order_h_workload_1(%arg0: index, %arg1: index, %arg2: index, %arg3: tensor<?x?x?x3xf64, {size_id = array<i64: 0, 2, 2, 3>}>) -> tensor<?x?x?x3xf64, {size_id = array<i64: 0, 2, 2, 3>, symmetry = #tc.symm<1~2>}> attributes {num_input_sizes = 3 : i64, workload = 2 : ui64} {
    %0 = tensor.empty(%arg0, %arg2, %arg2) : tensor<?x?x?x3xf64, {size_id = array<i64: 0, 2, 2, 3>, symmetry = #tc.symm<1~2>}>
    %1 = linalg.generic {indexing_maps = [#map1, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg3, %arg3 : tensor<?x?x?x3xf64, {size_id = array<i64: 0, 2, 2, 3>}>, tensor<?x?x?x3xf64, {size_id = array<i64: 0, 2, 2, 3>}>) outs(%0 : tensor<?x?x?x3xf64, {size_id = array<i64: 0, 2, 2, 3>, symmetry = #tc.symm<1~2>}>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %2 = arith.addf %in, %in_0 : f64
      linalg.yield %2 : f64
    } -> tensor<?x?x?x3xf64, {size_id = array<i64: 0, 2, 2, 3>, symmetry = #tc.symm<1~2>}>
    tc.return %1 : tensor<?x?x?x3xf64, {size_id = array<i64: 0, 2, 2, 3>, symmetry = #tc.symm<1~2>}>
  }
  func.func @first_order_h(%arg0: index {func.name = "batches"}, %arg1: index {func.name = "P"}, %arg2: index {func.name = "N"}, %arg3: tensor<?x?x3x?xf64, {size_id = array<i64: 0, 1, 3, 2>}> {func.name = "gradient_basis_wave"}, %arg4: tensor<?x?x3x3xf64, {size_id = array<i64: 0, 1, 3, 3>}> {func.name = "first_order_gradient_rho"}, %arg5: tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 3>}> {func.name = "gradient_rho"}, %arg6: tensor<?x?xf64, {size_id = array<i64: 0, 1>}> {func.name = "ptab"}, %arg7: tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 3>}> {func.name = "grid_coord"}, %arg8: tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 3>}> {func.name = "v_hartree_gradient"}, %arg9: tensor<?x?xf64, {size_id = array<i64: 0, 1>}> {func.name = "v2rho2"}, %arg10: tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 3>}> {func.name = "first_order_rho"}, %arg11: tensor<?x?xf64, {size_id = array<i64: 0, 1>}> {func.name = "v2rhosigma"}, %arg12: tensor<?x?xf64, {size_id = array<i64: 0, 1>}> {func.name = "v2sigma2"}, %arg13: tensor<?x?xf64, {size_id = array<i64: 0, 1>}> {func.name = "vsigma"}, %arg14: tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}> {func.name = "wave"}) -> (tensor<?x?x?x3xf64, {size_id = array<i64: 0, 2, 2, 3>}> {func.name = "H"}) attributes {num_input_indices = 0 : i64, num_input_sizes = 3 : i64, num_input_tensors = 12 : i64, sizes_lower_bound = array<i64: 256, 150, 256, 3>, sizes_upper_bound = array<i64: 768, 250, 768, 3>, wirtebacks = [-1], workload = 1 : ui64} {
    %0 = call @first_order_h_workload_0(%arg0, %arg1, %arg2, %arg14, %arg7, %arg8, %arg9, %arg3, %arg13, %arg5, %arg4, %arg12, %arg10, %arg6, %arg11) : (index, index, index, tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}>, tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 3>}>, tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 3>}>, tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, tensor<?x?x3x?xf64, {size_id = array<i64: 0, 1, 3, 2>}>, tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 3>}>, tensor<?x?x3x3xf64, {size_id = array<i64: 0, 1, 3, 3>}>, tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 3>}>, tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, tensor<?x?xf64, {size_id = array<i64: 0, 1>}>) -> tensor<?x?x?x3xf64, {size_id = array<i64: 0, 1, 2, 3>}>
    %1 = "tblas.batch.gemm"(%arg14, %0) {lhs_trans = true, num_K = 1 : i64, num_M = 1 : i64, num_N = 2 : i64, num_batch = 1 : i64, rhs_trans = false} : (tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}>, tensor<?x?x?x3xf64, {size_id = array<i64: 0, 1, 2, 3>}>) -> tensor<?x?x?x3xf64, {size_id = array<i64: 0, 2, 2, 3>}>
    %2 = call @first_order_h_workload_1(%arg0, %arg1, %arg2, %1) : (index, index, index, tensor<?x?x?x3xf64, {size_id = array<i64: 0, 2, 2, 3>}>) -> tensor<?x?x?x3xf64, {size_id = array<i64: 0, 2, 2, 3>, symmetry = #tc.symm<1~2>}>
    tc.return %2 : tensor<?x?x?x3xf64, {size_id = array<i64: 0, 2, 2, 3>, symmetry = #tc.symm<1~2>}>
  }
}
