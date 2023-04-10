#!python
import os
import sys
root_dir = os.path.abspath('..')
sys.path.insert(0, root_dir)
from test_tvm import test_tvm
from tvm import te, topi, auto_scheduler


@auto_scheduler.register_workload
def first_order_h(s0, s1, s2):
    # gradient_basis_wave
    t3 = te.placeholder((s0, s1, 3, s2,), dtype="float64", name="t3")
    # first_order_gradient_rho
    t4 = te.placeholder((s0, s1, 3, 3,), dtype="float64", name="t4")
    # gradient_rho
    t5 = te.placeholder((s0, s1, 3,), dtype="float64", name="t5")
    # ptab
    t6 = te.placeholder((s0, s1,), dtype="float64", name="t6")
    # grid_coord
    t7 = te.placeholder((s0, s1, 3,), dtype="float64", name="t7")
    # v_hartree_gradient
    t8 = te.placeholder((s0, s1, 3,), dtype="float64", name="t8")
    # v2rho2
    t9 = te.placeholder((s0, s1,), dtype="float64", name="t9")
    # first_order_rho
    t10 = te.placeholder((s0, s1, 3,), dtype="float64", name="t10")
    # v2rhosigma
    t11 = te.placeholder((s0, s1,), dtype="float64", name="t11")
    # v2sigma2
    t12 = te.placeholder((s0, s1,), dtype="float64", name="t12")
    # vsigma
    t13 = te.placeholder((s0, s1,), dtype="float64", name="t13")
    # wave
    t14 = te.placeholder((s0, s1, s2,), dtype="float64", name="t14")
    cst0 = 0.5
    var0 = te.compute((s0, s1, 3,), lambda *ls: cst0 * t6[ls[0]][ls[1]] * t7[ls[0]][ls[1]][ls[2]], name="var0")
    var1 = te.compute((s0, s1, 3,), lambda *ls: cst0 * t6[ls[0]][ls[1]] * t8[ls[0]][ls[1]][ls[2]], name="var1")
    var2 = te.compute((s0, s1,), lambda *ls: t6[ls[0]][ls[1]] * t9[ls[0]][ls[1]], name="var2")
    var3 = te.compute((s0, s1, 3,), lambda *ls: cst0 * t10[ls[0]][ls[1]][ls[2]] * var2[ls[0]][ls[1]], name="var3")
    var4 = te.compute((s0, s1, 3, 3,), lambda *ls: t5[ls[0]][ls[1]][ls[2]] * t4[ls[0]][ls[1]][ls[2]][ls[3]], name="var4")
    var5 = topi.sum(var4, axis=[2])
    var6 = te.compute((s0, s1,), lambda *ls: t6[ls[0]][ls[1]] * t11[ls[0]][ls[1]], name="var6")
    var7 = te.compute((s0, s1, 3,), lambda *ls: cst0 * var5[ls[0]][ls[1]][ls[2]] * var6[ls[0]][ls[1]], name="var7")
    var8 = te.compute((s0, s1, 3,), lambda *ls: var0[ls[0]][ls[1]][ls[2]] + var1[ls[0]][ls[1]][ls[2]] + var3[ls[0]][ls[1]][ls[2]] + var7[ls[0]][ls[1]][ls[2]], name="var8")
    var9 = te.compute((s0, s1, s2, 3,), lambda *ls: t14[ls[0]][ls[1]][ls[2]] * var8[ls[0]][ls[1]][ls[3]], name="var9")
    var10 = te.compute((s0, s1,), lambda *ls: t6[ls[0]][ls[1]] * t12[ls[0]][ls[1]], name="var10")
    var11 = te.compute((s0, s1, 3,), lambda *ls: var5[ls[0]][ls[1]][ls[2]] * var10[ls[0]][ls[1]], name="var11")
    var12 = te.compute((s0, s1, 3,), lambda *ls: t10[ls[0]][ls[1]][ls[2]] * var6[ls[0]][ls[1]], name="var12")
    var13 = te.compute((s0, s1, 3,), lambda *ls: var11[ls[0]][ls[1]][ls[2]] + var12[ls[0]][ls[1]][ls[2]], name="var13")
    var14 = te.compute((s0, s1,), lambda *ls: t6[ls[0]][ls[1]] * t13[ls[0]][ls[1]], name="var14")
    var15 = te.compute((s0, s1, 3, 3,), lambda *ls: t4[ls[0]][ls[1]][ls[2]][ls[3]] * var14[ls[0]][ls[1]], name="var15")
    var16 = te.compute((s0, s1, 3, 3,), lambda *ls: t5[ls[0]][ls[1]][ls[2]] * var13[ls[0]][ls[1]][ls[3]], name="var16")
    var17 = te.compute((s0, s1, 3, 3,), lambda *ls: var15[ls[0]][ls[1]][ls[2]][ls[3]] + var16[ls[0]][ls[1]][ls[2]][ls[3]], name="var17")
    var18 = te.compute((s0, s1, 3, s2, 3,), lambda *ls: t3[ls[0]][ls[1]][ls[2]][ls[3]] * var17[ls[0]][ls[1]][ls[2]][ls[4]], name="var18")
    var19 = topi.sum(var18, axis=[2])
    var20 = te.compute((s0, s1, s2, 3,), lambda *ls: var9[ls[0]][ls[1]][ls[2]][ls[3]] + var19[ls[0]][ls[1]][ls[2]][ls[3]], name="var20")
    var21 = te.compute((s0, s1, s2, s2, 3,), lambda *ls: t14[ls[0]][ls[1]][ls[2]] * var20[ls[0]][ls[1]][ls[3]][ls[4]], name="var21")
    var22 = topi.sum(var21, axis=[1])
    var23 = te.compute((s0, s2, s2, 3,), lambda *ls: var22[ls[0]][ls[1]][ls[2]][ls[3]] + var22[ls[0]][ls[2]][ls[1]][ls[3]], name="var23")
    return [t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, var23]


s0 = 1280
s1 = 192
s2 = 256
test_tvm('first_order_h', args=[s0, s1, s2, ])
