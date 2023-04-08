@auto_scheduler.register_workload
def first_order_r(s0, s1):
    t2 = te.placeholder((s0, s1,), dtype="float64", name="t2")
    t3 = te.placeholder((s1, s1,), dtype="float64", name="t3")
    var0 = te.compute((s0, s1, s1,), lambda *ls: t2[ls[0]][ls[1]] * t3[ls[2]][ls[1]], name="var0")
    a0 = te.reduce_axis((0, s1), name="a0")
    var1 = te.compute((s0, s1,), lambda *ls: te.sum(var0[ls[0]][a0][ls[1]], axis=a0), name="var1")
    var2 = te.compute((s0, s1,), lambda *ls: t2[ls[0]][ls[1]] * var1[ls[0]][ls[1]], name="var2")
    a1 = te.reduce_axis((0, s1), name="a1")
    var3 = te.compute((s0,), lambda *ls: te.sum(var2[ls[0]][a1], axis=a1), name="var3")
    return [t2, t3, var3]
