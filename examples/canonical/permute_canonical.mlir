%1 = tc.permute(%0) {axesmap = [(0, 1)]} 
    :  tensor<?x?xf64> -> tensor<?x?xf64>

%1 = tc.permute(%0) {axesmap = [(1, 0)]} 
    :  tensor<?x?xf64, {symmetry=<0~1>}>
    -> tensor<?x?xf64, {symmetry=<0~1>}>

