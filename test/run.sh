#!/bin/bash

set -e

pushd test
./emit_mlir.sh
./emit_tvm.sh
./emit_scop_c.sh
./emit_relay.sh
./emit_blas.sh
popd
