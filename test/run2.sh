#!/bin/bash

files=(
    "test/dielectric/First_Order_Gradient_R.tvm.py"
    "test/dielectric/First_Order_H.tvm.py"
    "test/dielectric/First_Order_R.tvm.py"
    "test/polarizability/First_Order_Gradient_R.tvm.py"
    "test/polarizability/First_Order_H.tvm.py"
)

for file_path in "${files[@]}"; do
    file_name=$(basename "$file_path" .py)
    dir_name=$(dirname "$file_path")

    pushd "$dir_name" > /dev/null
    log_file="${file_name}.log.txt"
    echo "In $dir_name"
    echo "Running $file_name, log: $dir_name/$log_file"
    python "$file_name.py" > "$log_file" 2>&1
    popd > /dev/null
done

