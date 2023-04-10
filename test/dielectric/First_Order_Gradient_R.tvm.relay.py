#!python
import tvm
import numpy as np
from tvm import relay, auto_scheduler
from tvm.contrib import graph_executor


@auto_scheduler.register_workload
def get_calculation(s0, s1, s2):
    dtype = "float64"
    shapes = {}
    shape = (s0, s1, 3, s2,)
    shapes["Gwave"] = shape
    t3 = relay.var("Gwave", shape=shape, dtype=dtype)
    shape = (s0, s2, s2,)
    shapes["DM"] = shape
    t4 = relay.var("DM", shape=shape, dtype=dtype)
    shape = (s0, s1, s2,)
    shapes["wave"] = shape
    t5 = relay.var("wave", shape=shape, dtype=dtype)
    term0 = relay.transpose(t4, axes=(0, 1, 2,))
    term1 = relay.transpose(t4, axes=(0, 2, 1,))
    var0 = relay.add(term0, term1)
    term0 = relay.expand_dims(t5, axis=3, num_newaxis=1)
    term0 = relay.transpose(term0, axes=(0, 1, 2, 3))
    term1 = relay.expand_dims(var0, axis=3, num_newaxis=1)
    term1 = relay.transpose(term1, axes=(0, 3, 2, 1))
    var1 = relay.multiply(term0, term1)
    var2 = relay.sum(var1, axis=[2])
    term0 = relay.expand_dims(var2, axis=3, num_newaxis=1)
    term0 = relay.transpose(term0, axes=(0, 1, 2, 3))
    term1 = relay.transpose(t3, axes=(0, 1, 3, 2))
    var3 = relay.multiply(term0, term1)
    var4 = relay.sum(var3, axis=[2])
    func = relay.Function([t3, t4, t5], var4)
    mod = tvm.IRModule.from_expr(func)
    return mod, shapes


s0 = 1280
s1 = 192
s2 = 256
dtype = "float64"
target = tvm.target.Target("llvm")
log_file = "temp.json"
print("Extract tasks...")
params = {}
mod, shapes = get_calculation(s0, s1, s2)
print(mod)
with tvm.transform.PassContext(opt_level=2):
    mod, params = relay.optimize(mod, target)
print(mod)
tasks, task_weights = auto_scheduler.extract_tasks(
    mod["main"], params, target, include_simple_tasks=True)

for idx, task in enumerate(tasks):
    print("========== Task %d  (workload key: %s) ==========" %
          (idx, task.workload_key))
    print(task.compute_dag)


def run_tuning():
    print("Begin tuning...")
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(
        repeat=1, min_repeat_ms=300, timeout=10)
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=200,  # change this to 20000 to achieve the best performance
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )
    tuner.tune(tune_option)


run_tuning()


print("Compile...")
with auto_scheduler.ApplyHistoryBest(log_file):
    with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
        lib = relay.build(mod, target=target, params=params)

dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](dev))
for key, shape in shapes.items:
    data = tvm.nd.array((np.random.uniform(size=shape)).astype(dtype))
    module.set_input(key, data)

print("Evaluate inference time cost...")
print(module.benchmark(dev, repeat=3, min_repeat_ms=500))
