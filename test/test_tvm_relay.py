import os
import sys
import numpy as np
from pprint import pprint
import tvm
from tvm import relay, auto_scheduler
from tvm.contrib import graph_executor

target = "llvm"
if len(sys.argv) > 1:
    target = sys.argv[1]
if target not in ["llvm", "cuda", "opencl"]:
    sys.exit(1)
if target == "llvm":
    target = "llvm -mcpu=skylake-avx512"
hardware_params = None
if target.startswith("llvm"):
    os.environ["TVM_NUM_THREADS"] = "1"
    hardware_params = auto_scheduler.HardwareParams(num_cores=1, target=target)

filename = os.path.basename(__file__)
log_file = filename + "." + target.split()[0] + ".json"


def create_task(mod):
    tasks, task_weights = auto_scheduler.extract_tasks(
        mod["main"], {}, target, include_simple_tasks=True, hardware_params=hardware_params)
    for idx, task in enumerate(tasks):
        print("========== Task %d  (workload key: %s) ==========" %
              (idx, task.workload_key))
        print(task.compute_dag)
    return tasks, task_weights


def tune_tasks(tasks, task_weights):
    print("Begin tuning...")
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(
        repeat=1, min_repeat_ms=300, timeout=10)
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=1024,
        num_measures_per_round=64,
        early_stopping=192,
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)])
    tuner.tune(tune_option)


def time_task(mod, shapes):
    print("Compile...")
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            lib = relay.build(mod, target=target, params={})
    dev = tvm.device(str(target), 0)
    module = graph_executor.GraphModule(lib["default"](dev))
    pprint(shapes)
    for key, shape in shapes.items():
        data = tvm.nd.array((np.random.uniform(size=shape)).astype("float64"))
        module.set_input(key, data)
    print("Evaluate inference time cost...")
    print(module.benchmark(dev, repeat=16, number=1, min_repeat_ms=500))
    print(flush=True)


def test_tvm_relay(mod, shapes):
    tasks, task_weights = create_task(mod)
    tune_tasks(tasks, task_weights)
    time_task(mod, shapes)
