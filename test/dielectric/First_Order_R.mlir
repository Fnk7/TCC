module {
  func.func @first_order_r(%arg0: index {func.name = "P"}, %arg1: index {func.name = "N"}, %arg2: tensor<?x?xf64, {size_id = array<i64: 0, 1>}> {func.name = "wave"}, %arg3: tensor<?x?xf64, {size_id = array<i64: 1, 1>}> {func.name = "DM"}) -> (tensor<?xf64, {size_id = array<i64: 0>}> {func.name = "R"}) attributes {num_input_indices = 0 : i64, num_input_sizes = 2 : i64, num_input_tensors = 2 : i64, sizes_lower_bound = array<i64: 180, 200>, sizes_upper_bound = array<i64: 200, 300>, wirtebacks = [-1]} {
    %0 = tc.mul(%arg2, %arg3) {axesmaps = [(0, 1), (2, 1)]} :(tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, tensor<?x?xf64, {size_id = array<i64: 1, 1>}>) -> tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 1>}>
    %1 = tc.contract(%0) {axes = [1]} : tensor<?x?x?xf64, {size_id = array<i64: 0, 1, 1>}> -> tensor<?x?xf64, {size_id = array<i64: 0, 1>}>
    %2 = tc.mul(%arg2, %1) {axesmaps = [(0, 1), (0, 1)]} :(tensor<?x?xf64, {size_id = array<i64: 0, 1>}>, tensor<?x?xf64, {size_id = array<i64: 0, 1>}>) -> tensor<?x?xf64, {size_id = array<i64: 0, 1>}>
    %3 = tc.contract(%2) {axes = [1]} : tensor<?x?xf64, {size_id = array<i64: 0, 1>}> -> tensor<?xf64, {size_id = array<i64: 0>}>
    tc.return %3 : tensor<?xf64, {size_id = array<i64: 0>}>
  }
}
