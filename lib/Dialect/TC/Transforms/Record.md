### Step 1
Any
=>
Add -> Contract -> Mul -> Input

### Step 1.1
Symm
Add
=>
Add -> Add

### Step 2
将Contract操作中的Axis按照可行顺序，将Contract拆开。
该步骤主要是每个Axis的Mul与之相关输入的集合之间比较。
如两个Axis对应的输入不相交，则可以拆分成两个并列的ContractOp。
如两个Axis对应的输入相同，则可以不拆分二者。
如某个Axis对应的输入是另一个的子集，则必定是其子节点。
最后，否则两个Axis对应的输入是相交的，且互不为子集，那么需要选择先后顺序。
同时，对Contract的输入的Mul的输入中，与Contract无关的Mul提出。
Contract -> Mul
=>
(Mul ->) Contract -> Mul ->Contract -> ... -> (Mul/Input)

### Step 3
合并加法中维度大小相同的Contract，为后面提取乘法做准备。
对于
```
sum((i), A(i) * sum((j), B(i,j))) +
sum((i,j), B(i,j) * C(i,j))
```
应该不予合并，尽管可能的合并形式
```
sum((i,j), (A(i) + C(i,j)) * B(i,j))
```
减少了N次（i的维度）乘法，但是增加了N*M次（i，j的维度）加法。
所以，合并时应该区别对待求和坐标 (i) 和 (i,j)。
同时，不应该区分对待 (i,j) 和 (j,i)。
Add -> Contract
=>
Add -> Contract -> Add

### Step 4
合并乘法
Add -> Mul
=>
(Add ->?) Mul -> Add (-> Mul?)
