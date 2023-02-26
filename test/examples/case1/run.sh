set -e
LLVM_HOME=`pwd`/../llvm-install/bin
TEST_CASE=case1
TEST_HOME=`pwd`/test/examples/${TEST_CASE}
rm -f ${TEST_HOME}/${TEST_CASE}.mlir ${TEST_HOME}/${TEST_CASE}.llvm.mlir ${TEST_HOME}/${TEST_CASE}.ll ${TEST_HOME}/a.out
./build/bin/tcc -emit=mlir-linalg ${TEST_HOME}/${TEST_CASE}.txt -opt-cse >${TEST_HOME}/${TEST_CASE}.mlir 2>/dev/null
${LLVM_HOME}/mlir-opt        \
    --convert-scf-to-cf      \
    --convert-cf-to-llvm     \
    --convert-arith-to-llvm  \
    --convert-func-to-llvm   \
    --convert-memref-to-llvm="use-generic-functions" \
    --reconcile-unrealized-casts \
    ${TEST_HOME}/${TEST_CASE}.mlir > ${TEST_HOME}/${TEST_CASE}.llvm.mlir
${LLVM_HOME}/mlir-translate --mlir-to-llvmir ${TEST_HOME}/${TEST_CASE}.llvm.mlir  > ${TEST_HOME}/${TEST_CASE}.ll
${LLVM_HOME}/clang++ ${TEST_HOME}/${TEST_CASE}.cpp ${TEST_HOME}/${TEST_CASE}.ll -o ${TEST_HOME}/a.out
${TEST_HOME}/a.out
