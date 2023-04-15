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
    shape = (s0, s1, s2,)
    shapes["wave"] = shape
    t3 = relay.var("wave", shape=shape, dtype=dtype)
    shape = (s0, s2, s2,)
    shapes["DM"] = shape
    t4 = relay.var("DM", shape=shape, dtype=dtype)
    term0 = relay.expand_dims(t3, axis=3, num_newaxis=1)
    term0 = relay.transpose(term0, axes=(0, 1, 2, 3,))
    term1 = relay.expand_dims(t4, axis=3, num_newaxis=1)
    term1 = relay.transpose(term1, axes=(0, 3, 2, 1,))
    var0 = relay.multiply(term0, term1)
    var1 = relay.sum(var0, axis=[2])
    term0 = relay.transpose(t3, axes=(0, 1, 2,))
    term1 = relay.transpose(var1, axes=(0, 1, 2,))
    var2 = relay.multiply(term0, term1)
    var3 = relay.sum(var2, axis=[2])
    func = relay.Function([t3, t4], var3)
    mod = tvm.IRModule.from_expr(func)
    return mod, shapes


s0 = 777
s1 = 192
s2 = 256
test_tvm_relay(get_calculation, s0, s1, s2)
