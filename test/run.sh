set -e
./build/bin/tcc -emit=tvm ./test/dielectric/First_Order_R.txt
./build/bin/tcc -emit=tvm ./test/dielectric/First_Order_Gradient_R.txt
./build/bin/tcc -emit=tvm ./test/dielectric/First_Order_H.txt
./build/bin/tcc -emit=tvm ./test/polarizability/First_Order_Gradient_R.txt
./build/bin/tcc -emit=tvm ./test/polarizability/First_Order_H.txt

