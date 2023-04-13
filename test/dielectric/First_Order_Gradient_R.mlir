module {
  func.func @first_order_gradient_r(%arg0: index {func.name = "P"}, %arg1: index {func.name = "N"}, %arg2: tensor<?x3x?xf64, {size_id = array<i64: 0, 2, 1>}> {func.name = "Gwave"}, %arg3: tensor<?x?xf64, {size_id = array<i64: 1, 1>}> {func.name = "DM"}, %arg4: tensor<?x?xf64, {size_id = array<i64: 0, 1>}> {func.name = "wave"}) -> (tensor<?x3xf64, {size_id = array<i64: 0, 2>}> {func.name = "GR"}) attributes {num_input_indices = 0 : i64, num_input_sizes = 2 : i64, num_input_tensors = 3 : i64, sizes_lower_bound = array<i64: 180, 200, 3>, sizes_upper_bound = array<i64: 200, 300, 3>, wirtebacks = [-1]} {
    %0 = tc.add(%arg3, %arg3) {axesmaps = [(0, 1), (1, 0)], signs = [+, +]} :(tensor<?x?xf64, {size_id = array<i64: 1, 1>}>, tensor<?x?xf64, {size_id = array<i64: 1, 1>}>) -> tensor<?x?xf64, {size_id = array<i64: 1, 1>, symmetry = #tc.symm<0~1>}>
    %1 = tc.mul(%arg4, %0) {axesmaps = [(0, 1), (2, 1)]} :(tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, tensor<?x?xf64, {size_id = array<i64: 1, 1>, symmetry = #tc.symm<0~1>}>) -> tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 1>}>
    %2 = tc.contract(%1) {axes = [1]} : tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 1>}> -> tensor<?x?xf64, {size_id = array<i64: 0, 1>}>
    %3 = tc.mul(%2, %arg2) {axesmaps = [(0, 1), (0, 2, 1)]} :(tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, tensor<?x3x?xf64, {size_id = array<i64: 0, 2, 1>}>) -> tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 2>}>
    %4 = tc.contract(%3) {axes = [1]} : tensor<?x?x3xf64, {size_id = array<i64: 0, 1, 2>}> -> tensor<?x3xf64, {size_id = array<i64: 0, 2>}>
    tc.return %4 : tensor<?x3xf64, {size_id = array<i64: 0, 2>}>
  }
}
