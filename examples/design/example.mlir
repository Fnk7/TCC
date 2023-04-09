module {
func.func @first_order_gradient_r(
              %arg0: index {func.name = "P"},
              %arg1: index {func.name = "N"},
              %arg2: tensor<?x3x?xf64, [0, 2, 1]> {func.name = "Gwave"},
              %arg3: tensor<?x?xf64, [1, 1]> {func.name = "DM"},
              %arg4: tensor<?x?xf64, [0, 1]> {func.name = "wave"}) 
                -> (tensor<3x?xf64, [2, 0]> {func.name = "GR"})
                  attributes {...} {
  %0 = tc.mul(%arg2, %arg3, %arg4) {axesmaps = [(0, 1, 2), (2, 3), (0, 3)]}
    :(tensor<?x3x?xf64,[0, 2, 1]>,tensor<?x?xf64,[1, 1]>,tensor<?x?xf64,[0, 1]>)
      -> tensor<?x3x?x?xf64, [0, 2, 1, 1]>
  %1 = tc.mul(%arg4, %arg3, %arg2) {axesmaps = [(0, 1), (1, 2), (0, 3, 2)]}
    :(tensor<?x?xf64,[0, 1]>,tensor<?x?xf64,[1, 1]>,tensor<?x3x?xf64, [0, 2, 1]>)
      -> tensor<?x?x?x3xf64, [0, 1, 1, 2]>
  %2 = tc.add(%0, %1) {axesmaps = [(0, 1, 2, 3), (0, 2, 3, 1)], signs = [+, +]}
    :(tensor<?x3x?x?xf64, [0, 2, 1, 1]>,tensor<?x?x?x3xf64, [0, 1, 1, 2]>)
      -> tensor<?x3x?x?xf64, [0, 2, 1, 1]>
  %3 = tc.contract(%2) {axes = [2, 3]} : tensor<?x3x?x?xf64, [0, 2, 1, 1]>
      -> tensor<?x3xf64, [0, 2]>
  %4 = tc.permute(%3) {axesmap = [(1, 0)]} : tensor<?x3xf64, [0, 2]>
      -> tensor<3x?xf64, [2, 0]>
  tc.return %4 : tensor<3x?xf64, [2, 0]>
}
}

