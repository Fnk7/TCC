#!python
import os
import tvm
from tvm import te, auto_scheduler, topi
import numpy as np

os.environ["TVM_NUM_THREADS"] = "72"

# generated
@auto_scheduler.register_workload
def tc_generated(s0, s1, s2):
    t3 = te.placeholder((s0, s1, 3,), dtype="float64", name="t3")
    t4 = te.placeholder((s0, s1, 3,), dtype="float64", name="t4")
    t5 = te.placeholder((s0, s1,), dtype="float64", name="t5")
    t6 = te.placeholder((s0, s1,), dtype="float64", name="t6")
    t7 = te.placeholder((s0, s1,), dtype="float64", name="t7")
    t8 = te.placeholder((s0, s1,), dtype="float64", name="t8")
    t9 = te.placeholder((s0, s1,), dtype="float64", name="t9")
    t10 = te.placeholder((s0, s1,), dtype="float64", name="t10")
    t11 = te.placeholder((s0, s1,), dtype="float64", name="t11")
    t12 = te.placeholder((s0, s1, s2,), dtype="float64", name="t12")
    t13 = te.placeholder((s0, s1, 3, s2,), dtype="float64", name="t13")
    cst0 = 0.5
    var0 = te.compute((s0, s1,), lambda *ls: cst0 * t6[ls[0]][ls[1]], name="var0")
    var1 = te.compute((s0, s1,), lambda *ls: t7[ls[0]][ls[1]] * t8[ls[0]][ls[1]], name="var1")
    var2 = te.compute((s0, s1,), lambda *ls: cst0 * var1[ls[0]][ls[1]], name="var2")
    var3 = te.compute((s0, s1, 3,), lambda *ls: t3[ls[0]][ls[1]][ls[2]] * t4[ls[0]][ls[1]][ls[2]], name="var3")
    a0 = te.reduce_axis((0, 3), name="a0")
    var4 = te.compute((s0, s1,), lambda *ls: te.sum(var3[ls[0]][ls[1]][a0], axis=a0), name="var4")
    var5 = te.compute((s0, s1,), lambda *ls: t9[ls[0]][ls[1]] * var4[ls[0]][ls[1]], name="var5")
    var6 = te.compute((s0, s1,), lambda *ls: cst0 * var5[ls[0]][ls[1]], name="var6")
    var7 = te.compute((s0, s1,), lambda *ls: var0[ls[0]][ls[1]] + var2[ls[0]][ls[1]] + var6[ls[0]][ls[1]], name="var7")
    var8 = te.compute((s0, s1,), lambda *ls: t5[ls[0]][ls[1]] * var7[ls[0]][ls[1]], name="var8")
    var9 = te.compute((s0, s1, s2,), lambda *ls: t12[ls[0]][ls[1]][ls[2]] * var8[ls[0]][ls[1]], name="var9")
    var10 = te.compute((s0, s1,), lambda *ls: t8[ls[0]][ls[1]] * t9[ls[0]][ls[1]], name="var10")
    var11 = te.compute((s0, s1,), lambda *ls: t10[ls[0]][ls[1]] * var4[ls[0]][ls[1]], name="var11")
    var12 = te.compute((s0, s1,), lambda *ls: var10[ls[0]][ls[1]] + var11[ls[0]][ls[1]], name="var12")
    var13 = te.compute((s0, s1,), lambda *ls: t5[ls[0]][ls[1]] * t11[ls[0]][ls[1]], name="var13")
    var14 = te.compute((s0, s1, 3,), lambda *ls: t4[ls[0]][ls[1]][ls[2]] * var13[ls[0]][ls[1]], name="var14")
    var15 = te.compute((s0, s1,), lambda *ls: t5[ls[0]][ls[1]] * var12[ls[0]][ls[1]], name="var15")
    var16 = te.compute((s0, s1, 3,), lambda *ls: t3[ls[0]][ls[1]][ls[2]] * var15[ls[0]][ls[1]], name="var16")
    var17 = te.compute((s0, s1, 3,), lambda *ls: var14[ls[0]][ls[1]][ls[2]] + var16[ls[0]][ls[1]][ls[2]], name="var17")
    var18 = te.compute((s0, s1, 3, s2,), lambda *ls: var17[ls[0]][ls[1]][ls[2]] * t13[ls[0]][ls[1]][ls[2]][ls[3]], name="var18")
    a1 = te.reduce_axis((0, 3), name="a1")
    var19 = te.compute((s0, s1, s2,), lambda *ls: te.sum(var18[ls[0]][ls[1]][a1][ls[2]], axis=a1), name="var19")
    var20 = te.compute((s0, s1, s2,), lambda *ls: var9[ls[0]][ls[1]][ls[2]] + var19[ls[0]][ls[1]][ls[2]], name="var20")
    var21 = te.compute((s0, s1, s2, s2,), lambda *ls: t12[ls[0]][ls[1]][ls[2]] * var20[ls[0]][ls[1]][ls[3]], name="var21")
    a2 = te.reduce_axis((0, s1), name="a2")
    var22 = te.compute((s0, s2, s2,), lambda *ls: te.sum(var21[ls[0]][a2][ls[1]][ls[2]], axis=a2), name="var22")
    var23 = te.compute((s0, s2, s2,), lambda *ls: var22[ls[0]][ls[1]][ls[2]] + var22[ls[0]][ls[2]][ls[1]], name="var23")
    var24 = te.compute((s0, s2, s2,), lambda *ls: var23[ls[0]][ls[2]][ls[1]], name="var24")
    return [t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, var24]

s0 = 1024
s1 = 192
s2 = 256
target = "cuda"
log_file = "tc_generated.json"
task = auto_scheduler.SearchTask(
    func=tc_generated, args=(s0, s1, s2), target=target)

print("Computational DAG:")
print(task.compute_dag)

tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=10,
    builder=auto_scheduler.LocalBuilder(timeout=100),
    runner=auto_scheduler.LocalRunner(timeout=100),
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    verbose=2,
)
# task.tune(tune_option)
sch, args = task.apply_best(log_file)

func = tvm.build(sch, args, target)

for m in func.imported_modules:
    print(m.get_source())

dev = tvm.device(target)

datas = []
for arg in args:
    shape = tuple(size.value for size in arg.shape)
    data = np.random.uniform(size=shape).astype(np.float64)
    datas.append(tvm.nd.array(data, device=dev))

evaluator = func.time_evaluator(func.entry_name, dev, repeat=1, min_repeat_ms=100)
print(
    "Magic time line x*x*x: %.3f ms"
    % (np.median(evaluator(*datas).results) * 1000)
)
