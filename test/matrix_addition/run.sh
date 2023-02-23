set -e
rm -f ./test/matrix_addition/temp.mlir ./test/matrix_addition/temp.llvm.mlir ./test/matrix_addition/temp.ll ./test/matrix_addition/a.out
./build/bin/tcc -emit=mlir-linalg ./test/matrix_addition/temp.txt -opt-cse 2>./test/matrix_addition/temp.mlir
../llvm-install/bin/mlir-opt --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm  --convert-func-to-llvm --convert-memref-to-llvm="use-generic-functions" --reconcile-unrealized-casts ./test/matrix_addition/temp.mlir > ./test/matrix_addition/temp.llvm.mlir
../llvm-install/bin/mlir-translate --mlir-to-llvmir ./test/matrix_addition/temp.llvm.mlir  > ./test/matrix_addition/temp.ll
../llvm-install/bin/clang++ ./test/matrix_addition/temp.cpp ./test/matrix_addition/temp.ll -o ./test/matrix_addition/a.out