#!python
import os
import sys
root_dir = os.path.abspath('..')
sys.path.insert(0, root_dir)
from test_tvm_relay import test_tvm_relay
import tvm
from tvm import relay


def get_calculation(s0, s1, s2):
    dtype = "float64"
    shapes = {}
    shape = (s0, s1, 3, s2,)
    shapes["gradient_basis_wave"] = shape
    t3 = relay.var("gradient_basis_wave", shape=shape, dtype=dtype)
    shape = (s0, s1, 3, 3,)
    shapes["first_order_gradient_rho"] = shape
    t4 = relay.var("first_order_gradient_rho", shape=shape, dtype=dtype)
    shape = (s0, s1, 3,)
    shapes["gradient_rho"] = shape
    t5 = relay.var("gradient_rho", shape=shape, dtype=dtype)
    shape = (s0, s1,)
    shapes["ptab"] = shape
    t6 = relay.var("ptab", shape=shape, dtype=dtype)
    shape = (s0, s1, 3,)
    shapes["grid_coord"] = shape
    t7 = relay.var("grid_coord", shape=shape, dtype=dtype)
    shape = (s0, s1, 3,)
    shapes["v_hartree_gradient"] = shape
    t8 = relay.var("v_hartree_gradient", shape=shape, dtype=dtype)
    shape = (s0, s1,)
    shapes["v2rho2"] = shape
    t9 = relay.var("v2rho2", shape=shape, dtype=dtype)
    shape = (s0, s1, 3,)
    shapes["first_order_rho"] = shape
    t10 = relay.var("first_order_rho", shape=shape, dtype=dtype)
    shape = (s0, s1,)
    shapes["v2rhosigma"] = shape
    t11 = relay.var("v2rhosigma", shape=shape, dtype=dtype)
    shape = (s0, s1,)
    shapes["v2sigma2"] = shape
    t12 = relay.var("v2sigma2", shape=shape, dtype=dtype)
    shape = (s0, s1,)
    shapes["vsigma"] = shape
    t13 = relay.var("vsigma", shape=shape, dtype=dtype)
    shape = (s0, s1, s2,)
    shapes["wave"] = shape
    t14 = relay.var("wave", shape=shape, dtype=dtype)
    cst0 = 0.5
    term0 = relay.expand_dims(t9, axis=2, num_newaxis=1)
    term0 = relay.transpose(term0, axes=(0, 1, 2,))
    term1 = relay.transpose(t10, axes=(0, 1, 2,))
    var0 = relay.multiply(term0, term1)
    term0 = relay.expand_dims(t5, axis=3, num_newaxis=1)
    term0 = relay.transpose(term0, axes=(0, 1, 2, 3,))
    term1 = relay.transpose(t4, axes=(0, 1, 2, 3,))
    var1 = relay.multiply(term0, term1)
    var2 = relay.sum(var1, axis=[2])
    term0 = relay.expand_dims(t11, axis=2, num_newaxis=1)
    term0 = relay.transpose(term0, axes=(0, 1, 2,))
    term1 = relay.transpose(var2, axes=(0, 1, 2,))
    var3 = relay.multiply(term0, term1)
    term0 = relay.transpose(t7, axes=(0, 1, 2,))
    term1 = relay.transpose(t8, axes=(0, 1, 2,))
    term2 = relay.transpose(var0, axes=(0, 1, 2,))
    term3 = relay.transpose(var3, axes=(0, 1, 2,))
    var4 = relay.add(term0, term1, term2, term3)
    term1 = relay.transpose(t6, axes=(0, 1,))
    var5 = relay.multiply(cst0, term1)
    term0 = relay.expand_dims(var5, axis=2, num_newaxis=1)
    term0 = relay.transpose(term0, axes=(0, 1, 2,))
    term1 = relay.transpose(var4, axes=(0, 1, 2,))
    var6 = relay.multiply(term0, term1)
    term0 = relay.expand_dims(t14, axis=3, num_newaxis=1)
    term0 = relay.transpose(term0, axes=(0, 1, 2, 3,))
    term1 = relay.expand_dims(var6, axis=3, num_newaxis=1)
    term1 = relay.transpose(term1, axes=(0, 1, 3, 2,))
    var7 = relay.multiply(term0, term1)
    term0 = relay.transpose(t6, axes=(0, 1,))
    term1 = relay.transpose(t12, axes=(0, 1,))
    var8 = relay.multiply(term0, term1)
    term0 = relay.transpose(var2, axes=(0, 1, 2,))
    term1 = relay.expand_dims(var8, axis=2, num_newaxis=1)
    term1 = relay.transpose(term1, axes=(0, 1, 2,))
    var9 = relay.multiply(term0, term1)
    term0 = relay.transpose(t6, axes=(0, 1,))
    term1 = relay.transpose(t11, axes=(0, 1,))
    var10 = relay.multiply(term0, term1)
    term0 = relay.transpose(t10, axes=(0, 1, 2,))
    term1 = relay.expand_dims(var10, axis=2, num_newaxis=1)
    term1 = relay.transpose(term1, axes=(0, 1, 2,))
    var11 = relay.multiply(term0, term1)
    term0 = relay.transpose(var9, axes=(0, 1, 2,))
    term1 = relay.transpose(var11, axes=(0, 1, 2,))
    var12 = relay.add(term0, term1)
    term0 = relay.transpose(t6, axes=(0, 1,))
    term1 = relay.transpose(t13, axes=(0, 1,))
    var13 = relay.multiply(term0, term1)
    term0 = relay.transpose(t4, axes=(0, 1, 2, 3,))
    term1 = relay.expand_dims(var13, axis=2, num_newaxis=2)
    term1 = relay.transpose(term1, axes=(0, 1, 2, 3,))
    var14 = relay.multiply(term0, term1)
    term0 = relay.expand_dims(t5, axis=3, num_newaxis=1)
    term0 = relay.transpose(term0, axes=(0, 1, 2, 3,))
    term1 = relay.expand_dims(var12, axis=3, num_newaxis=1)
    term1 = relay.transpose(term1, axes=(0, 1, 3, 2,))
    var15 = relay.multiply(term0, term1)
    term0 = relay.transpose(var14, axes=(0, 1, 2, 3,))
    term1 = relay.transpose(var15, axes=(0, 1, 2, 3,))
    var16 = relay.add(term0, term1)
    term0 = relay.expand_dims(t3, axis=4, num_newaxis=1)
    term0 = relay.transpose(term0, axes=(0, 1, 2, 3, 4,))
    term1 = relay.expand_dims(var16, axis=4, num_newaxis=1)
    term1 = relay.transpose(term1, axes=(0, 1, 2, 4, 3,))
    var17 = relay.multiply(term0, term1)
    var18 = relay.sum(var17, axis=[2])
    term0 = relay.transpose(var7, axes=(0, 1, 2, 3,))
    term1 = relay.transpose(var18, axes=(0, 1, 2, 3,))
    var19 = relay.add(term0, term1)
    term0 = relay.expand_dims(t14, axis=3, num_newaxis=2)
    term0 = relay.transpose(term0, axes=(0, 1, 2, 3, 4,))
    term1 = relay.expand_dims(var19, axis=4, num_newaxis=1)
    term1 = relay.transpose(term1, axes=(0, 1, 4, 2, 3,))
    var20 = relay.multiply(term0, term1)
    var21 = relay.sum(var20, axis=[1])
    term0 = relay.transpose(var21, axes=(0, 1, 2, 3,))
    term1 = relay.transpose(var21, axes=(0, 2, 1, 3,))
    var22 = relay.add(term0, term1)
    func = relay.Function([t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14], var22)
    mod = tvm.IRModule.from_expr(func)
    return mod, shapes


s0 = 777
s1 = 192
s2 = 256
test_tvm_relay(get_calculation, s0, s1, s2)
