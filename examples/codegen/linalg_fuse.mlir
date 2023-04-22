#map0 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
module {
func.func @first_order_gradient_r(%inputs) attributes {...} {
  %cst = arith.constant 0.000000e+00 : f64
  %0 = tensor.empty(%arg0, %arg1) : tensor<?x?xf64>
  %1 = linalg.fill ins(%cst : f64) outs(%0 : tensor<?x?xf64>) -> tensor<?x?xf64>
  %2 = linalg.generic {indexing_maps = [#map0, #map1, #map2, #map3], iterator_types = ["parallel", "reduction", "parallel"]} ins(%arg4, %arg3, %arg3 : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>) outs(%1 : tensor<?x?xf64>) {
  ^bb0(%in: f64, %in_0: f64, %in_1: f64, %out: f64):
    %6 = arith.addf %in_0, %in_1 : f64
    %7 = arith.mulf %in, %6 : f64
    %8 = arith.addf %7, %out : f64
    linalg.yield %8 : f64
  } -> tensor<?x?xf64>
  %3 = tensor.empty(%arg0) : tensor<?x3xf64>
  %4 = linalg.fill ins(%cst : f64) outs(%3 : tensor<?x3xf64>) -> tensor<?x3xf64>
  %5 = linalg.generic {indexing_maps = [#map0, #map4, #map3], iterator_types = ["parallel", "reduction", "parallel"]} ins(%2, %arg2 : tensor<?x?xf64>, tensor<?x3x?xf64>) outs(%4 : tensor<?x3xf64>) {
  ^bb0(%in: f64, %in_0: f64, %out: f64):
    %6 = arith.mulf %in, %in_0 : f64
    %7 = arith.addf %6, %out : f64
    linalg.yield %7 : f64
  } -> tensor<?x3xf64>
  return %5 : tensor<?x3xf64>
}
}


