### TC Dialect
下面是 Dialect 的 Op 定义和 Op 的规范形式定义。

#### Op
TC Dialect 目前 Op （操作）有 8 个，是 MulOp, AddOp, ContractOp, PermuteOp, SliceOp, DiagonalOp, ConstantOp, NegOp。这些操作的结果都是 MLIR 内置的张量 tensor 数据类型。

其中 MulOp, AddOp, ContractOp 是主要主要操作，计算的计算量主要在这三个算子上，优化在这上面进行。

SliceOp, ConstantOp, NegOp 主要是生成 MLIR 代码时的切片数据或常量数据或负数的表示。PermuteOp 主要是生成 MLIR 时对张量的重排，以符合写回结果，以及优化过程中的零时表示。 DiagonalOp 主要用于优化后代码使用对称矩阵对角线数据。

##### AddOp
加法，主要是 elementwise 的加法，但也可先扩充输入张量的维度后进行 elementwise 加法。

```
A(i,k,j) = B(j,k) + C(i,j)
```

输入：两个或以上的张量
属性：axes_maps : `array<array<i64>>`, signs : `array<bool>`
例子：
```
%A = tc.add(%B, %C) {axes_maps = [[2,1],[0,2]], signs = [0,0]}
```

对于每个输入，都有一个 `axes_maps` 中相关联的 `axes_map`。`axes_map` 是该输入张量与结果张量的映射。例子的 DSL 对应的 MLIR 代码中，第一个输入 `%B` 对应的 `axes_map` 是 `[2,1]`，表示输入 `B`索引 `j,k` 在结果 `A` 中的位置；同样，`axes_map [0,2]` 表示输入 `C`索引 `i,j` 在结果 `A` 中的位置。
同时，在加法操作中 `signs` 属性表示对应的输入是加法还是加法。

#### MulOp
乘法，将所有输入进行维度扩充后，进行 elementwise 的乘法。

```
Mul0(i,j,p) = W(i,p) * DM(i,j) * W(j,p)
%Mul0 = tc.mul(%W, %DM, %W) {axes_maps = [[0,2],[0,1],[1,2]]}
```

输入和属性与 AddOp 相同，但是没有 `signs` 这个属性。

#### ContractOp
张量求和，将张量的某些维度的数据全部求和。

```
A(i,j) = sum((k,l), B(i,k,j,l))
```

输入：一个张量
属性：axes : `array<i64>`
例子：
```
%A = tc.contract(%B) {axes = [1,3]}
```

属性 `axes` 是需要被求和的维度，例子中 `axes = [1,3]` 表示 `B` 的第二个和第三个维度的数据求和。

#### PermuteOp
张量维度重排。

```
A(i,j,k) = B(k,i,j)
```

输入：一个张量
属性：axes_map : `array<i64>`

属性 `axes_map` 与 AddOp 中一个输入对应的 `axes_map` 相同。

#### 其他
ConstantOp
```
A(i) = 0.5 * B(i)
=>
%cst0 = tc.constant() {value = 0.5} : () -> tensor<f64,[]>
%A = tc.mul(%cst0, %B) {axes_maps = [[],[0]]}
```

NegOp
```
A(i,j) = -B(i,j)
=>
%A = tc.neg(%B)
```

SliceOp
```
A(i,j) = sum((k), B(i,2,k), C(k,j))
=>
%cst2 = arith.constant() {value = 2} : () -> index
%sli0 = tc.slice(%B, %cst2) {axes = [1]} 
%mul0 = tc.mul(%sli0, %C) {axes_maps = [0,2],[2,1]}
%A = tc.contract(%mul0) {axes = [2]}
```

### Canonicalization
#### 例子
```
H(i,j) = sum((p), W(p,i) * pf(p) * W(p,j) + 
                  sum((c),GW(p,i,c) * pf2(p,c)) * W(p,j) + 
                  W(p,i) * sum((c),pf2(p,c) * GW(p,j,c)))
```

方案：先算乘法，然后加法，然后求和，最后加法。如果某个加法的两个输入中，求和操作的维度相同，先加法，后求和，如下面的 Add1 后 Sum2，否则先求和，再加法，如Sum1，Sum2 后加法求 H。
```
Mul1(p,i,j) = W(p,i) * pf1(p) * W(p,j)
Mul2(p,i,j,c) = GW(p,i,c) * pf2(p,c) * W(p,j);
Add1(p,i,j,c) = Mul2(p,i,j,c) + Mul2(p,j,i,c)
Sum1(i,j) = sum((p), Mul1(p,i,j))
Sum2(i,j) = sum((p,c), Add1(p,i,j,c))
H(i,j) = Sum1(i,j) + Sum2(i,j)
```

#### 规范形式
##### AddOp
输入顺序需要按照张量的维度信息排序，属性 `axes_maps` 化为标准形式。
输入中只能有一个 ConstantOp 的结果。
输入不能有 AddOp, PermuteOp, NegOp 的结果。
输入是 ContractOp 的结果，且维度相同，这两个 ContractOp 求和的维度不同。

属性 `axes_maps` 规范化如：
```
// A(i,j,k) = B(j,k) + C(i,k)
%A = tc.add(%B, %C) : {axes_maps = [[1,2],[0,2]], signs = [0,0]}
=>
// D(j,k,i) = B(j,k) + C(i,k)
// A(i,j,k) = D(j,k,i)
%D = tc.add(%B, %C) : {axes_maps = [[0,1],[2,1]], signs = [0,0]}
%A = tc.permute(%A) : {axes_map = [1,2,0]}
```

输入中只能有一个 ConstantOp 的结果：
```
%cst0 = tc.constant() {value = 1.2}
%cst1 = tc.constant() {value = 3.4}
%A = tc.add(%cst0, %cst1, %B) {axes_maps = [[],[],[0,1,2]], signs = [0,0,0]}
=>
%cst2 = tc.constant() {value = 4.6}
%A = tc.add(%cst2, %B) {axes_maps = [[],[0,1,2]], signs = [0,0]}
```

输入不能有 AddOp, PermuteOp, NegOp 的结果:
可以分别着重修改输入的个数，属性 `axes_maps` 和属性 `signs`，将这折叠起来。

将要实现的 Pattern
```c++
// Self: AddReorderPattern, AddAttrCanonicalizePattern
// Mul:
// Add: AddAddFoldPattern
// Contract: AddContractSwapPattern
// Permute: AddPermuteFoldPattern
// Others
// AddConstantFoldPattern
// AddNegFoldPattern
```

##### MulOp
输入顺序需要按照张量的维度信息排序，属性 `axes_maps` 化为标准形式。
输入中只能有一个 ConstantOp 的结果。
输入不能有 MulOp, ContractOp, AddOp, PermuteOp, NegOp 的结果。

对于第一点和第二点，处理方式与 AddOp 相同。
对于第三点，MulOp, PermuteOp 的处理与 AddOp 相同。
对于 MulOp 与 AddOp，应该将表达式展开，如下：
```
b * (a + c) + c * d
// or
a * b + c * (b + d)
=>
a * b + b * c + c * d
```

将要实现的 Pattern
```c++
// Self: MulReorderPattern, MulAttrCanonicalizePattern
// Mul: MulMulFoldPattern
// Add: MulAddSwapPattern
// Contract: MulContractSwapPattern
// Permute: MulPermuteFoldPattern
// Others
// MulConstantFoldPattern
// MulNegSwapPattern
```

##### ContractOp
属性 `axes` 化为标准形式。
输入不能是 PermuteOp 和 NegOp 的结果。
如果输入是加法，且加法中包含求和，则交换顺序。

输入不能是 PermuteOp, NegOp 的结果，交换两个算子的顺序：
```
A(i,j,k) = B(k,j,i)
C(i,j) = sum((k), A(i,j,k))
=>
D(j,i) = sum((k), B(k,j,i))
C(i,j) = D(j,i)
```

将要实现的 Pattern
```c++
// Mul:
// Add: ContractAddSwapPattern
// Contract: ContractContractFoldPattern
// Permute: ContractPermuteSwapPattern
// Others
// ContractNegSwapPattern
```

##### PermuteOp
输入不能是 PermuteOp 的结果。
PermuteOp 的结果与输入应该不同。
PermuteOp 的结果只能被 func Dialcet 的 ReturnOp 使用。

将要实现的 Pattern
```c++
// PermuteOp
// Self: PermuteFoldPattern
// Mul, Add, Contract:
// Permute: PermutePermuteFoldPattern
// Others
```

##### NegOp
会被 AddOp 消除。

将要实现的 Pattern
```c++
// NegOp
// Mul, Add, Contract:
// Permute: NegPermuteSwapPattern
// Others
// NegNegFoldPattern
```

##### SliceOp
输入只能是函数参数。

##### 其他
ConstantOp 即规范形式，DiagonalOp 不支持从 DSL 生成。
