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
    shape = (s0, s2, s2, 3,)
    shapes["DM"] = shape
    t3 = relay.var("DM", shape=shape, dtype=dtype)
    shape = (s0, s1, 3, s2,)
    shapes["gradient_basis_wave"] = shape
    t4 = relay.var("gradient_basis_wave", shape=shape, dtype=dtype)
    shape = (s0, s1, s2,)
    shapes["wave"] = shape
    t5 = relay.var("wave", shape=shape, dtype=dtype)
    term0 = relay.transpose(t3, axes=(0, 1, 2, 3,))
    term1 = relay.transpose(t3, axes=(0, 2, 1, 3,))
    var0 = relay.add(term0, term1)
    term0 = relay.expand_dims(t5, axis=3, num_newaxis=2)
    term0 = relay.transpose(term0, axes=(0, 1, 2, 3, 4,))
    term1 = relay.expand_dims(var0, axis=4, num_newaxis=1)
    term1 = relay.transpose(term1, axes=(0, 4, 1, 2, 3,))
    var1 = relay.multiply(term0, term1)
    var2 = relay.sum(var1, axis=[2])
    term0 = relay.expand_dims(var2, axis=4, num_newaxis=1)
    term0 = relay.transpose(term0, axes=(0, 1, 2, 3, 4,))
    term1 = relay.expand_dims(t4, axis=4, num_newaxis=1)
    term1 = relay.transpose(term1, axes=(0, 1, 3, 4, 2,))
    var3 = relay.multiply(term0, term1)
    var4 = relay.sum(var3, axis=[2])
    var5 = te.compute((s0, s1, 3, 3,), lambda *ls: var4[ls[0]][ls[1]][ls[3]][ls[2]], name="var5")
    func = relay.Function([t3, t4, t5], var5)
    mod = tvm.IRModule.from_expr(func)
    return mod, shapes


s0 = 777
s1 = 192
s2 = 256
test_tvm_relay(get_calculation, s0, s1, s2)
