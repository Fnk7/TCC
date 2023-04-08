#!python
from ..python.common import *
from tvm import te, topi, auto_scheduler


@auto_scheduler.register_workload
def first_order_r(s0, s1, s2):
    # wave
    t3 = te.placeholder((s0, s1, s2,), dtype="float64", name="t3")
    # DM
    t4 = te.placeholder((s0, s2, s2,), dtype="float64", name="t4")
    var0 = te.compute((s0, s1, s2, s2,), lambda *ls: t3[ls[0]][ls[1]][ls[2]] * t4[ls[0]][ls[3]][ls[2]], name="var0")
    var1 = topi.sum(var0, axis=[2])
    var2 = te.compute((s0, s1, s2,), lambda *ls: t3[ls[0]][ls[1]][ls[2]] * var1[ls[0]][ls[1]][ls[2]], name="var2")
    var3 = topi.sum(var2, axis=[2])
    return [t3, t4, var3]


def getTask(target):
    s0 = 768
    s1 = 200
    s2 = 256
    log_file = 'first_order_r' + '_' + target + '.json'
    task = auto_scheduler.SearchTask(func=first_order_r, args=(s0, s1, s2, ), target=target)
    return task, log_file


for target in targets:
    task, log_file = getTask(target)
    print_compute_dag(task)
    tune_task(task, log_file)
    time_task(task, log_file, target)
