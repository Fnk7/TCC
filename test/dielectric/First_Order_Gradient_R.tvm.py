@auto_scheduler.register_workload
def first_order_gradient_r(s0, s1):
    t2 = te.placeholder((s0, 3, s1,), dtype="float64", name="t2")
    t3 = te.placeholder((s1, s1,), dtype="float64", name="t3")
    t4 = te.placeholder((s0, s1,), dtype="float64", name="t4")
    var0 = te.compute((s1, s1,), lambda *ls: t3[ls[0]][ls[1]] + t3[ls[1]][ls[0]], name="var0")
    var1 = te.compute((s0, s1, s1,), lambda *ls: t4[ls[0]][ls[1]] * var0[ls[2]][ls[1]], name="var1")
    a0 = te.reduce_axis((0, s1), name="a0")
    var2 = te.compute((s0, s1,), lambda *ls: te.sum(var1[ls[0]][a0][ls[1]], axis=a0), name="var2")
    var3 = te.compute((s0, s1, 3,), lambda *ls: var2[ls[0]][ls[1]] * t2[ls[0]][ls[2]][ls[1]], name="var3")
    a1 = te.reduce_axis((0, s1), name="a1")
    var4 = te.compute((s0, 3,), lambda *ls: te.sum(var3[ls[0]][a1][ls[1]], axis=a1), name="var4")
    return [t2, t3, t4, var4]
