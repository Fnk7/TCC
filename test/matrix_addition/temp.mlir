module {
  func.func @CT(%arg0: index, %arg1: index, %arg2: memref<?x?xf64> {func.name = "B"}, %arg3: memref<?x?xf64> {func.name = "C"}, %arg4: memref<?x?xf64>) attributes {llvm.emit_c_interface, num_input_indices = 0 : i64, num_input_sizes = 2 : i64, num_input_tensors = 2 : i64, wirtebacks = [-1]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = memref.dim %arg2, %c0 : memref<?x?xf64>
    %dim_0 = memref.dim %arg2, %c1 : memref<?x?xf64>
    scf.for %arg5 = %c0 to %dim step %c1 {
      scf.for %arg6 = %c0 to %dim_0 step %c1 {
        %0 = memref.load %arg2[%arg5, %arg6] : memref<?x?xf64>
        %1 = memref.load %arg3[%arg5, %arg6] : memref<?x?xf64>
        %2 = arith.addf %0, %1 : f64
        memref.store %2, %arg4[%arg5, %arg6] : memref<?x?xf64>
      }
    }
    return
  }
}
