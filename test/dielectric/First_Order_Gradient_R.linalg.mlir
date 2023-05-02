#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
module {
  func.func @first_order_gradient_r(%arg0: index {func.name = "batches"}, %arg1: index {func.name = "P"}, %arg2: index {func.name = "N"}, %arg3: tensor<?x?x3x?xf64, {size_id = array<i64: 0, 1, 3, 2>}> {func.name = "Gwave"}, %arg4: tensor<?x?x?xf64, {size_id = array<i64: 0, 2, 2>}> {func.name = "DM"}, %arg5: tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}> {func.name = "wave"}) -> (tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 3>}> {func.name = "GR"}) attributes {num_input_indices = 0 : i64, num_input_sizes = 3 : i64, num_input_tensors = 3 : i64, sizes_lower_bound = array<i64: 256, 150, 256, 3>, sizes_upper_bound = array<i64: 768, 250, 768, 3>, wirtebacks = [-1]} {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = tensor.empty(%arg0, %arg1, %arg2) : tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}>
    %1 = linalg.fill ins(%cst : f64) outs(%0 : tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}>) -> tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}>
    %2 = linalg.generic {indexing_maps = [#map0, #map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%arg5, %arg4, %arg4 : tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}>, tensor<?x?x?xf64, {size_id = array<i64: 0, 2, 2>}>, tensor<?x?x?xf64, {size_id = array<i64: 0, 2, 2>}>) outs(%1 : tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %out: f64):
      %6 = arith.addf %in_0, %in_1 : f64
      %7 = arith.mulf %in, %6 : f64
      %8 = arith.addf %7, %out : f64
      linalg.yield %8 : f64
    } -> tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}>
    %3 = tensor.empty(%arg0, %arg1) : tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 3>}>
    %4 = linalg.fill ins(%cst : f64) outs(%3 : tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 3>}>) -> tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 3>}>
    %5 = linalg.generic {indexing_maps = [#map0, #map4, #map3], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%2, %arg3 : tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}>, tensor<?x?x3x?xf64, {size_id = array<i64: 0, 1, 3, 2>}>) outs(%4 : tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 3>}>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %6 = arith.mulf %in, %in_0 : f64
      %7 = arith.addf %6, %out : f64
      linalg.yield %7 : f64
    } -> tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 3>}>
    tc.return %5 : tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 3>}>
  }
}
