#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map7 = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
module {
  func.func @first_order_h(%arg0: index {func.name = "batches"}, %arg1: index {func.name = "P"}, %arg2: index {func.name = "N"}, %arg3: tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 3>}> {func.name = "gradient_rho"}, %arg4: tensor<?x?x3x?xf64, {size_id = array<i64: 0, 1, 3, 2>}> {func.name = "gradient_basis_wave"}, %arg5: tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 3>}> {func.name = "first_order_gradient_rho"}, %arg6: tensor<?x?xf64, {size_id = array<i64: 0, 1>}> {func.name = "ptab"}, %arg7: tensor<?x?xf64, {size_id = array<i64: 0, 1>}> {func.name = "first_order_potential"}, %arg8: tensor<?x?xf64, {size_id = array<i64: 0, 1>}> {func.name = "v2rho2"}, %arg9: tensor<?x?xf64, {size_id = array<i64: 0, 1>}> {func.name = "first_order_rho"}, %arg10: tensor<?x?xf64, {size_id = array<i64: 0, 1>}> {func.name = "v2rhosigma"}, %arg11: tensor<?x?xf64, {size_id = array<i64: 0, 1>}> {func.name = "v2sigma2"}, %arg12: tensor<?x?xf64, {size_id = array<i64: 0, 1>}> {func.name = "vsigma"}, %arg13: tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}> {func.name = "wave"}) -> (tensor<?x?x?xf64, {size_id = array<i64: 0, 2, 2>}> {func.name = "H"}) attributes {num_input_indices = 0 : i64, num_input_sizes = 3 : i64, num_input_tensors = 11 : i64, sizes_lower_bound = array<i64: 256, 150, 256, 3>, sizes_upper_bound = array<i64: 768, 250, 768, 3>, wirtebacks = [-1]} {
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 5.000000e-01 : f64
    %0 = tensor.empty(%arg0, %arg1) : tensor<?x?xf64, {size_id = array<i64: 0, 1>}>
    %1 = linalg.fill ins(%cst : f64) outs(%0 : tensor<?x?xf64, {size_id = array<i64: 0, 1>}>) -> tensor<?x?xf64, {size_id = array<i64: 0, 1>}>
    %2 = linalg.generic {indexing_maps = [#map0, #map0, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg3, %arg5 : tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 3>}>, tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 3>}>) outs(%1 : tensor<?x?xf64, {size_id = array<i64: 0, 1>}>) {
    ^bb0(%in: f64, %in_1: f64, %out: f64):
      %11 = arith.mulf %in, %in_1 : f64
      %12 = arith.addf %11, %out : f64
      linalg.yield %12 : f64
    } -> tensor<?x?xf64, {size_id = array<i64: 0, 1>}>
    %3 = tensor.empty(%arg0, %arg1, %arg2) : tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}>
    %4 = linalg.fill ins(%cst : f64) outs(%3 : tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}>) -> tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}>
    %5 = linalg.generic {indexing_maps = [#map2, #map3, #map3, #map2, #map3, #map3, #map3, #map3, #map4, #map5], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%arg5, %arg6, %arg12, %arg3, %arg9, %arg10, %arg11, %2, %arg4 : tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 3>}>, tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 3>}>, tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, tensor<?x?x3x?xf64, {size_id = array<i64: 0, 1, 3, 2>}>) outs(%4 : tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}>) {
    ^bb0(%in: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %in_6: f64, %in_7: f64, %in_8: f64, %out: f64):
      %11 = arith.mulf %in_6, %in_7 : f64
      %12 = arith.mulf %in_4, %in_5 : f64
      %13 = arith.addf %12, %11 : f64
      %14 = arith.mulf %in_1, %13 : f64
      %15 = arith.mulf %in_3, %14 : f64
      %16 = arith.mulf %in_1, %in_2 : f64
      %17 = arith.mulf %in, %16 : f64
      %18 = arith.addf %17, %15 : f64
      %19 = arith.mulf %18, %in_8 : f64
      %20 = arith.addf %19, %out : f64
      linalg.yield %20 : f64
    } -> tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}>
    %6 = tensor.empty(%arg0, %arg2, %arg2) : tensor<?x?x?xf64, {size_id = array<i64: 0, 2, 2>}>
    %7 = linalg.fill ins(%cst : f64) outs(%6 : tensor<?x?x?xf64, {size_id = array<i64: 0, 2, 2>}>) -> tensor<?x?x?xf64, {size_id = array<i64: 0, 2, 2>}>
    %8 = linalg.generic {indexing_maps = [#map2, #map5, #map3, #map3, #map3, #map3, #map3, #map3, #map5, #map6], iterator_types = ["parallel", "reduction", "parallel", "parallel"]} ins(%arg13, %arg13, %arg6, %arg7, %arg8, %arg9, %arg10, %2, %5 : tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}>, tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}>, tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}>) outs(%7 : tensor<?x?x?xf64, {size_id = array<i64: 0, 2, 2>}>) {
    ^bb0(%in: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %in_6: f64, %in_7: f64, %in_8: f64, %out: f64):
      %11 = arith.mulf %in_6, %in_7 : f64
      %12 = arith.mulf %in_4, %in_5 : f64
      %13 = arith.addf %in_3, %12 : f64
      %14 = arith.addf %13, %11 : f64
      %15 = arith.mulf %in_2, %14 : f64
      %16 = arith.mulf %15, %cst_0 : f64
      %17 = arith.mulf %in_1, %16 : f64
      %18 = arith.addf %17, %in_8 : f64
      %19 = arith.mulf %in, %18 : f64
      %20 = arith.addf %19, %out : f64
      linalg.yield %20 : f64
    } -> tensor<?x?x?xf64, {size_id = array<i64: 0, 2, 2>}>
    %9 = tensor.empty(%arg0, %arg2, %arg2) : tensor<?x?x?xf64, {size_id = array<i64: 0, 2, 2>, symmetry = #tc.symm<1~2>}>
    %10 = linalg.generic {indexing_maps = [#map0, #map7, #map0], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8, %8 : tensor<?x?x?xf64, {size_id = array<i64: 0, 2, 2>}>, tensor<?x?x?xf64, {size_id = array<i64: 0, 2, 2>}>) outs(%9 : tensor<?x?x?xf64, {size_id = array<i64: 0, 2, 2>, symmetry = #tc.symm<1~2>}>) {
    ^bb0(%in: f64, %in_1: f64, %out: f64):
      %11 = arith.addf %in, %in_1 : f64
      linalg.yield %11 : f64
    } -> tensor<?x?x?xf64, {size_id = array<i64: 0, 2, 2>, symmetry = #tc.symm<1~2>}>
    tc.return %10 : tensor<?x?x?xf64, {size_id = array<i64: 0, 2, 2>, symmetry = #tc.symm<1~2>}>
  }
}
