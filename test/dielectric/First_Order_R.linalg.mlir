#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @first_order_r(%arg0: index {func.name = "batches"}, %arg1: index {func.name = "P"}, %arg2: index {func.name = "N"}, %arg3: tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}> {func.name = "wave"}, %arg4: tensor<?x?x?xf64, {size_id = array<i64: 0, 2, 2>}> {func.name = "DM"}) -> (tensor<?x?xf64, {size_id = array<i64: 0, 1>}> {func.name = "R"}) attributes {num_input_indices = 0 : i64, num_input_sizes = 3 : i64, num_input_tensors = 2 : i64, sizes_lower_bound = array<i64: 256, 150, 256>, sizes_upper_bound = array<i64: 768, 250, 768>, wirtebacks = [-1]} {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = tensor.empty(%arg0, %arg1, %arg2) : tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}>
    %1 = linalg.fill ins(%cst : f64) outs(%0 : tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}>) -> tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}>
    %2 = linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%arg3, %arg4 : tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}>, tensor<?x?x?xf64, {size_id = array<i64: 0, 2, 2>}>) outs(%1 : tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %6 = arith.mulf %in, %in_0 : f64
      %7 = arith.addf %6, %out : f64
      linalg.yield %7 : f64
    } -> tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}>
    %3 = tensor.empty(%arg0, %arg1) : tensor<?x?xf64, {size_id = array<i64: 0, 1>}>
    %4 = linalg.fill ins(%cst : f64) outs(%3 : tensor<?x?xf64, {size_id = array<i64: 0, 1>}>) -> tensor<?x?xf64, {size_id = array<i64: 0, 1>}>
    %5 = linalg.generic {indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg3, %2 : tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}>, tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 2>}>) outs(%4 : tensor<?x?xf64, {size_id = array<i64: 0, 1>}>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %6 = arith.mulf %in, %in_0 : f64
      %7 = arith.addf %6, %out : f64
      linalg.yield %7 : f64
    } -> tensor<?x?xf64, {size_id = array<i64: 0, 1>}>
    tc.return %5 : tensor<?x?xf64, {size_id = array<i64: 0, 1>}>
  }
}
