@auto_scheduler.register_workload
def first_order_gradient_r(s0, s1):
    t2 = te.placeholder((s0, s0, 3,), dtype="float64", name="t2")
    t3 = te.placeholder((s1, 3, s0,), dtype="float64", name="t3")
    t4 = te.placeholder((s1, s0,), dtype="float64", name="t4")
    var0 = te.compute((s0, s0, 3,), lambda *ls: t2[ls[0]][ls[1]][ls[2]] + t2[ls[1]][ls[0]][ls[2]], name="var0")
    var1 = te.compute((s0, s0, 3, s1,), lambda *ls: var0[ls[0]][ls[1]][ls[2]] * t4[ls[3]][ls[0]], name="var1")
    a0 = te.reduce_axis((0, s0), name="a0")
    var2 = te.compute((s0, 3, s1,), lambda *ls: te.sum(var1[a0][ls[0]][ls[1]][ls[2]], axis=a0), name="var2")
    var3 = te.compute((s0, 3, s1, 3,), lambda *ls: var2[ls[0]][ls[1]][ls[2]] * t3[ls[2]][ls[3]][ls[0]], name="var3")
    a1 = te.reduce_axis((0, s0), name="a1")
    var4 = te.compute((3, s1, 3,), lambda *ls: te.sum(var3[a1][ls[0]][ls[1]][ls[2]], axis=a1), name="var4")
    var5 = te.compute((s1, 3, 3,), lambda *ls: var4[ls[2]][ls[0]][ls[1]], name="var5")
    return [t2, t3, t4, var5]
