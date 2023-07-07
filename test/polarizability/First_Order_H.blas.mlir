#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map6 = affine_map<(d0, d1, d2) -> (d0)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map10 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map11 = affine_map<(d0) -> ()>
#map12 = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
module {
  func.func @first_order_h_workload_0(%arg0: index, %arg1: index, %arg2: tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, %arg3: tensor<?x3xf64, {size_id = array<i64: 0, 2>}>, %arg4: tensor<?x3xf64, {size_id = array<i64: 0, 2>}>, %arg5: tensor<?xf64, {size_id = array<i64: 0>}>, %arg6: tensor<?x3x?xf64, {size_id = array<i64: 0, 2, 1>}>, %arg7: tensor<?xf64, {size_id = array<i64: 0>}>, %arg8: tensor<?x3xf64, {size_id = array<i64: 0, 2>}>, %arg9: tensor<?x3x3xf64, {size_id = array<i64: 0, 2, 2>}>, %arg10: tensor<?xf64, {size_id = array<i64: 0>}>, %arg11: tensor<?x3xf64, {size_id = array<i64: 0, 2>}>, %arg12: tensor<?xf64, {size_id = array<i64: 0>}>, %arg13: tensor<?xf64, {size_id = array<i64: 0>}>) -> tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 2>}> attributes {num_input_sizes = 2 : i64, sizes_lower_bound = array<i64: 150, 256, 3>, sizes_upper_bound = array<i64: 250, 768, 3>, workload = 2 : ui64} {
    %cst = arith.constant 5.000000e-01 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %0 = tensor.empty(%arg0) : tensor<?xf64, {size_id = array<i64: 0>}>
    %1 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%arg12, %arg13 : tensor<?xf64, {size_id = array<i64: 0>}>, tensor<?xf64, {size_id = array<i64: 0>}>) outs(%0 : tensor<?xf64, {size_id = array<i64: 0>}>) {
    ^bb0(%in: f64, %in_1: f64, %out: f64):
      %29 = arith.mulf %in, %in_1 : f64
      linalg.yield %29 : f64
    } -> tensor<?xf64, {size_id = array<i64: 0>}>
    %2 = tensor.empty(%arg0) : tensor<?x3xf64, {size_id = array<i64: 0, 2>}>
    %3 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg11, %1 : tensor<?x3xf64, {size_id = array<i64: 0, 2>}>, tensor<?xf64, {size_id = array<i64: 0>}>) outs(%2 : tensor<?x3xf64, {size_id = array<i64: 0, 2>}>) {
    ^bb0(%in: f64, %in_1: f64, %out: f64):
      %29 = arith.mulf %in, %in_1 : f64
      linalg.yield %29 : f64
    } -> tensor<?x3xf64, {size_id = array<i64: 0, 2>}>
    %4 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%arg12, %arg10 : tensor<?xf64, {size_id = array<i64: 0>}>, tensor<?xf64, {size_id = array<i64: 0>}>) outs(%0 : tensor<?xf64, {size_id = array<i64: 0>}>) {
    ^bb0(%in: f64, %in_1: f64, %out: f64):
      %29 = arith.mulf %in, %in_1 : f64
      linalg.yield %29 : f64
    } -> tensor<?xf64, {size_id = array<i64: 0>}>
    %5 = tensor.empty(%arg0) : tensor<?x3x3xf64, {size_id = array<i64: 0, 2, 2>}>
    %6 = linalg.generic {indexing_maps = [#map3, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg8, %arg9 : tensor<?x3xf64, {size_id = array<i64: 0, 2>}>, tensor<?x3x3xf64, {size_id = array<i64: 0, 2, 2>}>) outs(%5 : tensor<?x3x3xf64, {size_id = array<i64: 0, 2, 2>}>) {
    ^bb0(%in: f64, %in_1: f64, %out: f64):
      %29 = arith.mulf %in, %in_1 : f64
      linalg.yield %29 : f64
    } -> tensor<?x3x3xf64, {size_id = array<i64: 0, 2, 2>}>
    %7 = linalg.fill ins(%cst_0 : f64) outs(%2 : tensor<?x3xf64, {size_id = array<i64: 0, 2>}>) -> tensor<?x3xf64, {size_id = array<i64: 0, 2>}>
    %8 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "reduction", "parallel"]} ins(%6 : tensor<?x3x3xf64, {size_id = array<i64: 0, 2, 2>}>) outs(%7 : tensor<?x3xf64, {size_id = array<i64: 0, 2>}>) {
    ^bb0(%in: f64, %out: f64):
      %29 = arith.addf %in, %out : f64
      linalg.yield %29 : f64
    } -> tensor<?x3xf64, {size_id = array<i64: 0, 2>}>
    %9 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%8, %4 : tensor<?x3xf64, {size_id = array<i64: 0, 2>}>, tensor<?xf64, {size_id = array<i64: 0>}>) outs(%2 : tensor<?x3xf64, {size_id = array<i64: 0, 2>}>) {
    ^bb0(%in: f64, %in_1: f64, %out: f64):
      %29 = arith.mulf %in, %in_1 : f64
      linalg.yield %29 : f64
    } -> tensor<?x3xf64, {size_id = array<i64: 0, 2>}>
    %10 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%9, %3 : tensor<?x3xf64, {size_id = array<i64: 0, 2>}>, tensor<?x3xf64, {size_id = array<i64: 0, 2>}>) outs(%2 : tensor<?x3xf64, {size_id = array<i64: 0, 2>}>) {
    ^bb0(%in: f64, %in_1: f64, %out: f64):
      %29 = arith.addf %in, %in_1 : f64
      linalg.yield %29 : f64
    } -> tensor<?x3xf64, {size_id = array<i64: 0, 2>}>
    %11 = linalg.generic {indexing_maps = [#map3, #map5, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg8, %10 : tensor<?x3xf64, {size_id = array<i64: 0, 2>}>, tensor<?x3xf64, {size_id = array<i64: 0, 2>}>) outs(%5 : tensor<?x3x3xf64, {size_id = array<i64: 0, 2, 2>}>) {
    ^bb0(%in: f64, %in_1: f64, %out: f64):
      %29 = arith.mulf %in, %in_1 : f64
      linalg.yield %29 : f64
    } -> tensor<?x3x3xf64, {size_id = array<i64: 0, 2, 2>}>
    %12 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%arg12, %arg7 : tensor<?xf64, {size_id = array<i64: 0>}>, tensor<?xf64, {size_id = array<i64: 0>}>) outs(%0 : tensor<?xf64, {size_id = array<i64: 0>}>) {
    ^bb0(%in: f64, %in_1: f64, %out: f64):
      %29 = arith.mulf %in, %in_1 : f64
      linalg.yield %29 : f64
    } -> tensor<?xf64, {size_id = array<i64: 0>}>
    %13 = linalg.generic {indexing_maps = [#map4, #map6, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %12 : tensor<?x3x3xf64, {size_id = array<i64: 0, 2, 2>}>, tensor<?xf64, {size_id = array<i64: 0>}>) outs(%5 : tensor<?x3x3xf64, {size_id = array<i64: 0, 2, 2>}>) {
    ^bb0(%in: f64, %in_1: f64, %out: f64):
      %29 = arith.mulf %in, %in_1 : f64
      linalg.yield %29 : f64
    } -> tensor<?x3x3xf64, {size_id = array<i64: 0, 2, 2>}>
    %14 = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%13, %11 : tensor<?x3x3xf64, {size_id = array<i64: 0, 2, 2>}>, tensor<?x3x3xf64, {size_id = array<i64: 0, 2, 2>}>) outs(%5 : tensor<?x3x3xf64, {size_id = array<i64: 0, 2, 2>}>) {
    ^bb0(%in: f64, %in_1: f64, %out: f64):
      %29 = arith.addf %in, %in_1 : f64
      linalg.yield %29 : f64
    } -> tensor<?x3x3xf64, {size_id = array<i64: 0, 2, 2>}>
    %15 = tensor.empty(%arg0, %arg1) : tensor<?x3x?x3xf64, {size_id = array<i64: 0, 2, 1, 2>}>
    %16 = linalg.generic {indexing_maps = [#map7, #map8, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg6, %14 : tensor<?x3x?xf64, {size_id = array<i64: 0, 2, 1>}>, tensor<?x3x3xf64, {size_id = array<i64: 0, 2, 2>}>) outs(%15 : tensor<?x3x?x3xf64, {size_id = array<i64: 0, 2, 1, 2>}>) {
    ^bb0(%in: f64, %in_1: f64, %out: f64):
      %29 = arith.mulf %in, %in_1 : f64
      linalg.yield %29 : f64
    } -> tensor<?x3x?x3xf64, {size_id = array<i64: 0, 2, 1, 2>}>
    %17 = tensor.empty(%arg0, %arg1) : tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 2>}>
    %18 = linalg.fill ins(%cst_0 : f64) outs(%17 : tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 2>}>) -> tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 2>}>
    %19 = linalg.generic {indexing_maps = [#map9, #map10], iterator_types = ["parallel", "reduction", "parallel", "parallel"]} ins(%16 : tensor<?x3x?x3xf64, {size_id = array<i64: 0, 2, 1, 2>}>) outs(%18 : tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 2>}>) {
    ^bb0(%in: f64, %out: f64):
      %29 = arith.addf %in, %out : f64
      linalg.yield %29 : f64
    } -> tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 2>}>
    %20 = linalg.generic {indexing_maps = [#map2, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg13, %8 : tensor<?xf64, {size_id = array<i64: 0>}>, tensor<?x3xf64, {size_id = array<i64: 0, 2>}>) outs(%2 : tensor<?x3xf64, {size_id = array<i64: 0, 2>}>) {
    ^bb0(%in: f64, %in_1: f64, %out: f64):
      %29 = arith.mulf %in, %in_1 : f64
      linalg.yield %29 : f64
    } -> tensor<?x3xf64, {size_id = array<i64: 0, 2>}>
    %21 = linalg.generic {indexing_maps = [#map2, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg5, %arg11 : tensor<?xf64, {size_id = array<i64: 0>}>, tensor<?x3xf64, {size_id = array<i64: 0, 2>}>) outs(%2 : tensor<?x3xf64, {size_id = array<i64: 0, 2>}>) {
    ^bb0(%in: f64, %in_1: f64, %out: f64):
      %29 = arith.mulf %in, %in_1 : f64
      linalg.yield %29 : f64
    } -> tensor<?x3xf64, {size_id = array<i64: 0, 2>}>
    %22 = linalg.generic {indexing_maps = [#map1, #map1, #map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg3, %arg4, %21, %20 : tensor<?x3xf64, {size_id = array<i64: 0, 2>}>, tensor<?x3xf64, {size_id = array<i64: 0, 2>}>, tensor<?x3xf64, {size_id = array<i64: 0, 2>}>, tensor<?x3xf64, {size_id = array<i64: 0, 2>}>) outs(%2 : tensor<?x3xf64, {size_id = array<i64: 0, 2>}>) {
    ^bb0(%in: f64, %in_1: f64, %in_2: f64, %in_3: f64, %out: f64):
      %29 = arith.addf %in, %in_1 : f64
      %30 = arith.addf %29, %in_2 : f64
      %31 = arith.addf %30, %in_3 : f64
      linalg.yield %31 : f64
    } -> tensor<?x3xf64, {size_id = array<i64: 0, 2>}>
    %23 = tensor.empty() : tensor<f64, {size_id = array<i64>}>
    %24 = linalg.fill ins(%cst : f64) outs(%23 : tensor<f64, {size_id = array<i64>}>) -> tensor<f64, {size_id = array<i64>}>
    %25 = linalg.generic {indexing_maps = [#map11, #map0, #map0], iterator_types = ["parallel"]} ins(%24, %arg12 : tensor<f64, {size_id = array<i64>}>, tensor<?xf64, {size_id = array<i64: 0>}>) outs(%0 : tensor<?xf64, {size_id = array<i64: 0>}>) {
    ^bb0(%in: f64, %in_1: f64, %out: f64):
      %29 = arith.mulf %in, %in_1 : f64
      linalg.yield %29 : f64
    } -> tensor<?xf64, {size_id = array<i64: 0>}>
    %26 = linalg.generic {indexing_maps = [#map2, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%25, %22 : tensor<?xf64, {size_id = array<i64: 0>}>, tensor<?x3xf64, {size_id = array<i64: 0, 2>}>) outs(%2 : tensor<?x3xf64, {size_id = array<i64: 0, 2>}>) {
    ^bb0(%in: f64, %in_1: f64, %out: f64):
      %29 = arith.mulf %in, %in_1 : f64
      linalg.yield %29 : f64
    } -> tensor<?x3xf64, {size_id = array<i64: 0, 2>}>
    %27 = linalg.generic {indexing_maps = [#map3, #map5, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg2, %26 : tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, tensor<?x3xf64, {size_id = array<i64: 0, 2>}>) outs(%17 : tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 2>}>) {
    ^bb0(%in: f64, %in_1: f64, %out: f64):
      %29 = arith.mulf %in, %in_1 : f64
      linalg.yield %29 : f64
    } -> tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 2>}>
    %28 = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%27, %19 : tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 2>}>, tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 2>}>) outs(%17 : tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 2>}>) {
    ^bb0(%in: f64, %in_1: f64, %out: f64):
      %29 = arith.addf %in, %in_1 : f64
      linalg.yield %29 : f64
    } -> tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 2>}>
    tc.return %28 : tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 2>}>
  }
  func.func @first_order_h_workload_1(%arg0: index, %arg1: index, %arg2: tensor<?x?x3xf64, {size_id = array<i64: 1, 1, 2>}>) -> tensor<?x?x3xf64, {size_id = array<i64: 1, 1, 2>, symmetry = #tc.symm<0~1>}> attributes {num_input_sizes = 2 : i64, sizes_lower_bound = array<i64: 150, 256, 3>, sizes_upper_bound = array<i64: 250, 768, 3>, workload = 2 : ui64} {
    %0 = tensor.empty(%arg1, %arg1) : tensor<?x?x3xf64, {size_id = array<i64: 1, 1, 2>, symmetry = #tc.symm<0~1>}>
    %1 = linalg.generic {indexing_maps = [#map4, #map12, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg2, %arg2 : tensor<?x?x3xf64, {size_id = array<i64: 1, 1, 2>}>, tensor<?x?x3xf64, {size_id = array<i64: 1, 1, 2>}>) outs(%0 : tensor<?x?x3xf64, {size_id = array<i64: 1, 1, 2>, symmetry = #tc.symm<0~1>}>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %2 = arith.addf %in, %in_0 : f64
      linalg.yield %2 : f64
    } -> tensor<?x?x3xf64, {size_id = array<i64: 1, 1, 2>, symmetry = #tc.symm<0~1>}>
    tc.return %1 : tensor<?x?x3xf64, {size_id = array<i64: 1, 1, 2>, symmetry = #tc.symm<0~1>}>
  }
  func.func @first_order_h(%arg0: index {func.name = "P"}, %arg1: index {func.name = "N"}, %arg2: tensor<?x3x?xf64, {size_id = array<i64: 0, 2, 1>}> {func.name = "gradient_basis_wave"}, %arg3: tensor<?x3x3xf64, {size_id = array<i64: 0, 2, 2>}> {func.name = "first_order_gradient_rho"}, %arg4: tensor<?x3xf64, {size_id = array<i64: 0, 2>}> {func.name = "gradient_rho"}, %arg5: tensor<?xf64, {size_id = array<i64: 0>}> {func.name = "ptab"}, %arg6: tensor<?x3xf64, {size_id = array<i64: 0, 2>}> {func.name = "grid_coord"}, %arg7: tensor<?x3xf64, {size_id = array<i64: 0, 2>}> {func.name = "v_hartree_gradient"}, %arg8: tensor<?xf64, {size_id = array<i64: 0>}> {func.name = "v2rho2"}, %arg9: tensor<?x3xf64, {size_id = array<i64: 0, 2>}> {func.name = "first_order_rho"}, %arg10: tensor<?xf64, {size_id = array<i64: 0>}> {func.name = "v2rhosigma"}, %arg11: tensor<?xf64, {size_id = array<i64: 0>}> {func.name = "v2sigma2"}, %arg12: tensor<?xf64, {size_id = array<i64: 0>}> {func.name = "vsigma"}, %arg13: tensor<?x?xf64, {size_id = array<i64: 0, 1>}> {func.name = "wave"}) -> (tensor<?x?x3xf64, {size_id = array<i64: 1, 1, 2>}> {func.name = "H"}) attributes {num_input_indices = 0 : i64, num_input_sizes = 2 : i64, num_input_tensors = 12 : i64, sizes_lower_bound = array<i64: 150, 256, 3>, sizes_upper_bound = array<i64: 250, 768, 3>, wirtebacks = [-1], workload = 1 : ui64} {
    %0 = call @first_order_h_workload_0(%arg0, %arg1, %arg13, %arg6, %arg7, %arg8, %arg2, %arg12, %arg4, %arg3, %arg11, %arg9, %arg5, %arg10) : (index, index, tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, tensor<?x3xf64, {size_id = array<i64: 0, 2>}>, tensor<?x3xf64, {size_id = array<i64: 0, 2>}>, tensor<?xf64, {size_id = array<i64: 0>}>, tensor<?x3x?xf64, {size_id = array<i64: 0, 2, 1>}>, tensor<?xf64, {size_id = array<i64: 0>}>, tensor<?x3xf64, {size_id = array<i64: 0, 2>}>, tensor<?x3x3xf64, {size_id = array<i64: 0, 2, 2>}>, tensor<?xf64, {size_id = array<i64: 0>}>, tensor<?x3xf64, {size_id = array<i64: 0, 2>}>, tensor<?xf64, {size_id = array<i64: 0>}>, tensor<?xf64, {size_id = array<i64: 0>}>) -> tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 2>}>
    %1 = "tblas.batch.gemm"(%arg13, %0) {lhs_trans = true, num_K = 1 : i64, num_M = 1 : i64, num_N = 2 : i64, num_batch = 0 : i64, rhs_trans = false} : (tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 2>}>) -> tensor<?x?x3xf64, {size_id = array<i64: 1, 1, 2>}>
    %2 = call @first_order_h_workload_1(%arg0, %arg1, %1) : (index, index, tensor<?x?x3xf64, {size_id = array<i64: 1, 1, 2>}>) -> tensor<?x?x3xf64, {size_id = array<i64: 1, 1, 2>, symmetry = #tc.symm<0~1>}>
    tc.return %2 : tensor<?x?x3xf64, {size_id = array<i64: 1, 1, 2>, symmetry = #tc.symm<0~1>}>
  }
}
