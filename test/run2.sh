#!/bin/bash

files=(
    "dielectric/First_Order_Gradient_R.tvm.py"
    "dielectric/First_Order_H.tvm.py"
    "dielectric/First_Order_R.tvm.py"
    "polarizability/First_Order_Gradient_R.tvm.py"
    "polarizability/First_Order_H.tvm.py"
)

targets=(
    "llvm"
    "cuda"
    "opencl"
)

for target in "${targets[@]}"; do
    for file_path in "${files[@]}"; do
        file_name=$(basename "$file_path" .py)
        dir_name=$(dirname "$file_path")
        pushd "$dir_name" > /dev/null
        log_file="${file_name}.${target}.log"
        err_file="${file_name}.${target}.err"
        echo "${file_name}.py ${target}"
        echo "    log: ${dir_name}/${log_file}"
        python "${file_name}.py" ${target} >"${log_file}" 2>"${err_file}"
        popd > /dev/null
    done
    echo ""
done
