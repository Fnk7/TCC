# TCC — Tensor Contraction Compiler

This is a small, experimental prototype from my master's thesis. It applies
linear-algebra transformations to reduce the computational cost of tensor
contractions in quantum-chemistry calculations. It is built on
[MLIR](https://mlir.llvm.org/) and is not a complete or production-ready tool.

## Overview

A linear tensor contraction has the form

$$R_{c,p} = \sum_{i,j} G_{p,c,i} \cdot D_{i,j} \cdot W_{p,j}$$

where the sum runs over the contracted indices i and j. Such expressions appear
in quantum-chemistry calculations. The focus here is the linear-algebra
side: rewriting these expressions (symmetry, factorization, contraction and
multiplication reordering, batching) to reduce the number of floating-point
operations. The
code-generation backends are experimental and used to inspect the lowered
result. The pieces are:

- a small text format for declaring tensor sizes and contraction expressions;
- a parser that builds a contraction tree and emits MLIR;
- a custom MLIR dialect (`tc`) for contraction operations and a few algebraic
  rewrites;
- a `tblas` dialect for tensor batch BLAS;
- experimental lowering and code-generation paths for C, TVM, and TVM
  Relay.

## Input Language

Tensors and their index sizes are declared first, then expressions are written
in an Einstein-summation-like syntax:

```
size P = (150, 250), N = (256, 768), C = (3, 3)
input Gwave(P, C, N), DM(N, N), wave(P, N)

GR(c, p) = sum((i, j),
               Gwave(p, c, i) * DM(i, j) * wave(p, j)
             + wave(p, i) * DM(i, j) * Gwave(p, c, j))
keep GR
```

- `size` declares an index together with its `[lower, upper]` bounds.
- `input` declares input tensors.
- `sum((i, j), ...)` sums the enclosed expression over the listed indices.
- `keep` marks a tensor as a result to be emitted.

The grammar is:

```
<indices>   ::= '(' <index> (',' <index>)* ')'
<expr>      ::= <term> ('+' <term>)*
<term>      ::= <factor> ('*' <factor>)*
<sum>       ::= "sum" '(' <indices> ',' <expr> ')'
<factor>    ::= <tensor> <indices> | <sum> | '(' <expr> ')'
<statement> ::= <tensor> <indices> '=' <expr>
```

## Compilation Pipeline

```
 input text
    │  Parser / ContractionTree / MLIRGen
    ▼
 tc dialect (contract, add, mul, permute, …)
    │  algebraic optimization (canonicalization, symmetry, factorization,
    │  multiplication order, contraction order, batching, workload split)
    ▼
 lowered dialects
    ├─ linalg ─────────────────────────────► C (SCoP-style)
    ├─ tblas ──► linalg ───────────────────► C (BLAS)
    └─ tc ─────────────────────────────────► TVM / TVM Relay expressions
```

1. **Front-end** (`lib/Parser`): parses the input text, builds a
   `ContractionTree` AST, and lowers it to MLIR in the `tc` dialect.
2. **Optimization** (`lib/Dialect/TC/Transforms`, `lib/Passes`):
   canonicalization, symmetry, factorization, multiplication and
   contraction reordering, batching (`add-batch`), and workload splitting.
3. **Lowering**: `tc` is lowered to `linalg` (`LowerTCToLinalg`) and to `tblas`
   (`LowerTCToTBLAS`).
4. **Code generation** (`lib/CodeGen`): emits C (`EmitCCode*.cpp`), TVM
   (`EmitTVM.cpp`), and TVM Relay (`EmitTVMRelay.cpp`).

The algebraic-optimization stage only reduces the number of floating-point
operations; the low-level, high-performance code is produced by the backends —
platform-optimized BLAS libraries, polyhedral code generation (SCoP, with LLVM
Polly for CPU and PPCG for GPU), and TVM auto-tuning.

Direct GPU code generation was something I did not manage: I did not know how
to split the work into grids and blocks, and in 2023 the affine path could not
produce high-performance code either. The tile-based IRs
available today are a much better fit for this kind of work, if they can
handle tensors with dynamic sizes.

## Dialects

### `tc` — Tensor Contraction

Operations: `tc.add`, `tc.mul`, `tc.contract`, `tc.permute`, `tc.constant`,
`tc.neg`, `tc.slice`, `tc.diagonal`, `tc.return`.

### `tblas` — Tensor Batch BLAS

Operations: `tblas.generic`, `tblas.batch.gemm`, `tblas.batch.gemv`,
`tblas.batch.dot`, `tblas.batch.syr2k`, `tblas.batch.syrk`,
`tblas.batch.syrkx`, `tblas.batch.trmm`.

## Building

### Prerequisites

- CMake ≥ 3.13, Ninja, and a C++17 compiler
- An installed LLVM/MLIR (with `clang` if you want to compile the emitted C)

### 1. Build LLVM/MLIR

TCC is an out-of-tree MLIR project, so LLVM/MLIR must be configured and
installed first. It was originally developed against **LLVM 16**:

```sh
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/path/to/llvm-install \
  -DLLVM_BUILD_UTILS=ON -DLLVM_INSTALL_UTILS=ON
ninja install
```

### 2. Build TCC

```sh
cmake -G Ninja /path/to/TCC \
  -DMLIR_DIR=/path/to/llvm-install/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm-install/lib/cmake/llvm \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/path/to/llvm-install/bin/clang++
ninja tcc
```

This produces the `tcc` driver at `build/bin/tcc`.

## Usage

```sh
tcc -emit=<action> [options] <input.txt>
```

`-emit` actions:

| Action         | Output                                        |
|----------------|-----------------------------------------------|
| `ast`          | AST dump                                      |
| `mlir`         | `tc`-dialect MLIR after optimization          |
| `mlir-linalg`  | MLIR after linalg lowering                    |
| `mlir-blas`    | MLIR after `tblas` lowering (+ C)             |
| `scop-c`       | SCoP-style C code                             |
| `tvm`          | TVM tensor-expression Python                  |
| `relay`        | TVM Relay Python                              |

Options:

| Option            | Meaning                                        |
|-------------------|------------------------------------------------|
| `-tc-no-opt`      | disable `tc`-dialect optimization              |
| `-linalg-fuse`    | fuse linalg generic parallel loops             |
| `-add-batch`      | add batch dimensions                           |
| `-lower-linalg`   | lower linalg to loops                          |
| `-allow-small-blas` | allow small contractions to become BLAS kernels |

Examples:

```sh
tcc -emit=mlir    test/dielectric/First_Order_Gradient_R.txt
tcc -emit=scop-c  -add-batch -linalg-fuse test/polarizability/First_Order_H.txt
tcc -emit=tvm     -add-batch              test/polarizability/First_Order_H.txt
tcc -emit=relay   -add-batch              test/polarizability/First_Order_H.txt
```

## Directory Layout

```
include/TCC/           public headers (dialects, passes, codegen, parser)
lib/
  Parser/              text parser, contraction tree, MLIR generation
  Dialect/
    TC/                tc dialect (IR, canonicalizations, transforms)
    TBLAS/             tblas dialect (IR, transforms)
  Passes/              lowering and optimization passes
  CodeGen/             C / TVM / TVM Relay emitters
tools/tcc/             the tcc command-line driver
test/                  regression inputs and golden outputs
```
