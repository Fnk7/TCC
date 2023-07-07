module {
  func.func @gemm(%arg0: index {func.name = "M"}, %arg1: tensor<?x64xf64, {size_id = array<i64: 0, 2>}> {func.name = "A"}, %arg2: tensor<64x256xf64, {size_id = array<i64: 2, 1>}> {func.name = "B"}) -> (tensor<?x256xf64, {size_id = array<i64: 0, 1>}> {func.name = "C"}) attributes {num_input_indices = 0 : i64, num_input_sizes = 1 : i64, num_input_tensors = 2 : i64, sizes_lower_bound = array<i64: 128, 256, 64>, sizes_upper_bound = array<i64: 256, 256, 64>, wirtebacks = [-1]} {
    %0 = tc.mul(%arg1, %arg2) {axesmaps = [(0, 1), (1, 2)]} :(tensor<?x64xf64, {size_id = array<i64: 0, 2>}>, tensor<64x256xf64, {size_id = array<i64: 2, 1>}>) -> tensor<?x64x256xf64, {size_id = array<i64: 0, 2, 1>}>
    %1 = tc.contract(%0) {axes = [1]} : tensor<?x64x256xf64, {size_id = array<i64: 0, 2, 1>}> -> tensor<?x256xf64, {size_id = array<i64: 0, 1>}>
    %2 = tc.permute(%1) {axesmap = [(0, 1)]} : tensor<?x256xf64, {size_id = array<i64: 0, 1>}> -> tensor<?x256xf64, {size_id = array<i64: 0, 1>}>
    tc.return %2 : tensor<?x256xf64, {size_id = array<i64: 0, 1>}>
  }
}
