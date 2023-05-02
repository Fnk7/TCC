#!/bin/bash

tcc="../build/bin/tcc"

files=(
    "dielectric/First_Order_Gradient_R.txt"
    "dielectric/First_Order_H.txt"
    "dielectric/First_Order_R.txt"
    "polarizability/First_Order_Gradient_R.txt"
    "polarizability/First_Order_H.txt"
)

for file_path in "${files[@]}"; do
    dir_name=$(dirname "$file_path")
    ${tcc} -emit=scop-c -add-batch -linalg-fuse "${file_path}"
    ${tcc} -emit=mlir-blas -add-batch -linalg-fuse "${file_path}"
done
