#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map6 = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
module {
  func.func @first_order_gradient_r(%arg0: index {func.name = "P"}, %arg1: index {func.name = "N"}, %arg2: tensor<?x3x?xf64> {func.name = "Gwave"}, %arg3: tensor<?x?xf64> {func.name = "DM"}, %arg4: tensor<?x?xf64> {func.name = "wave"}) -> (tensor<?x3xf64> {func.name = "GR"}) attributes {num_input_indices = 0 : i64, num_input_sizes = 2 : i64, num_input_tensors = 3 : i64, sizes_lower_bound = array<i64: 180, 200, 3>, sizes_upper_bound = array<i64: 200, 300, 3>, wirtebacks = [-1]} {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = tensor.empty(%arg1, %arg1) : tensor<?x?xf64>
    %1 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg3, %arg3 : tensor<?x?xf64>, tensor<?x?xf64>) outs(%0 : tensor<?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %12 = arith.addf %in, %in_0 : f64
      linalg.yield %12 : f64
    } -> tensor<?x?xf64>

    %2 = tensor.empty(%arg0, %arg1, %arg1) : tensor<?x?x?xf64>
    %3 = linalg.generic {
            indexing_maps = [#map2, #map3, #map4], 
            iterator_types = ["parallel", "parallel", "parallel"]} 
            ins(%arg4, %1 : tensor<?x?xf64>, tensor<?x?xf64>) 
            outs(%2 : tensor<?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %12 = arith.mulf %in, %in_0 : f64
      linalg.yield %12 : f64
    } -> tensor<?x?x?xf64>

    %4 = tensor.empty(%arg0, %arg1) : tensor<?x?xf64>
    %5 = linalg.fill ins(%cst : f64) outs(%4 : tensor<?x?xf64>) -> tensor<?x?xf64>
    %6 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "reduction", "parallel"]} ins(%3 : tensor<?x?x?xf64>) outs(%5 : tensor<?x?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %12 = arith.addf %in, %out : f64
      linalg.yield %12 : f64
    } -> tensor<?x?xf64>
    %7 = tensor.empty(%arg0, %arg1) : tensor<?x?x3xf64>
    %8 = linalg.generic {indexing_maps = [#map2, #map6, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%6, %arg2 : tensor<?x?xf64>, tensor<?x3x?xf64>) outs(%7 : tensor<?x?x3xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %12 = arith.mulf %in, %in_0 : f64
      linalg.yield %12 : f64
    } -> tensor<?x?x3xf64>
    %9 = tensor.empty(%arg0) : tensor<?x3xf64>
    %10 = linalg.fill ins(%cst : f64) outs(%9 : tensor<?x3xf64>) -> tensor<?x3xf64>
    %11 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "reduction", "parallel"]} ins(%8 : tensor<?x?x3xf64>) outs(%10 : tensor<?x3xf64>) {
    ^bb0(%in: f64, %out: f64):
      %12 = arith.addf %in, %out : f64
      linalg.yield %12 : f64
    } -> tensor<?x3xf64>
    return %11 : tensor<?x3xf64>
  }
}