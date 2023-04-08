import tvm
from tvm import te, topi, auto_scheduler
@auto_scheduler.register_workload
def first_order_gradient_r(s0, s1, s2):
    # Gwave
    t3 = te.placeholder((s0, s1, 3, s2,), dtype="float64", name="t3")
    # DM
    t4 = te.placeholder((s0, s2, s2,), dtype="float64", name="t4")
    # wave
    t5 = te.placeholder((s0, s1, s2,), dtype="float64", name="t5")
    var0 = te.compute((s0, s2, s2,), lambda *ls: t4[ls[0]][ls[1]][ls[2]] + t4[ls[0]][ls[2]][ls[1]], name="var0")
    var1 = te.compute((s0, s1, s2, s2,), lambda *ls: t5[ls[0]][ls[1]][ls[2]] * var0[ls[0]][ls[3]][ls[2]], name="var1")
    var2 = topi.sum(var1, axis=[2])
    var3 = te.compute((s0, s1, s2, 3,), lambda *ls: var2[ls[0]][ls[1]][ls[2]] * t3[ls[0]][ls[1]][ls[3]][ls[2]], name="var3")
    var4 = topi.sum(var3, axis=[2])
    return [t3, t4, t5, var4]
