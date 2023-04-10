import os
import sys
import tvm
import numpy as np
from tvm import auto_scheduler

target = None
if len(sys.argv) > 1:
    target = sys.argv[1]
if target is None:
    target = "cuda"
if target not in ["llvm", "cuda", "opencl"]:
    sys.exit(1)
if target == "llvm":
    target = "llvm -mcpu=skylake-avx512"
hardware_params = None
if target.startswith("llvm"):
    os.environ["TVM_NUM_THREADS"] = "1"
    hardware_params = auto_scheduler.HardwareParams(num_cores=1, target=target)


def create_task(func, args):
    task = auto_scheduler.SearchTask(
        func=func, args=args, target=target, hardware_params=hardware_params)
    print("Computational DAG:")
    print(task.compute_dag, flush=True)
    return task


def tune_task(task, log_file):
    if target.startswith("llvm"):
        runner = auto_scheduler.LocalRunner(timeout=10, number=72, repeat=1)
    else:
        runner = auto_scheduler.LocalRunner(timeout=100, number=1, repeat=1)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=1024,
        early_stopping=128,
        num_measures_per_round=32,
        builder=auto_scheduler.LocalBuilder(timeout=45),
        runner=runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2)
    task.tune(tune_option)


def time_task(task, log_file):
    task.print_best(log_file)
    sch, args = task.apply_best(log_file)
    func = tvm.build(sch, args, target)
    for m in func.imported_modules:
        print(m.get_source())
    dev = tvm.device(target)
    datas = []
    for arg in args:
        shape = tuple(size.value for size in arg.shape)
        data = np.random.uniform(size=shape).astype(np.float64)
        datas.append(tvm.nd.array(data, device=dev))
    evaluator = func.time_evaluator(
        func.entry_name, dev, repeat=10, min_repeat_ms=100)
    print("0v0 Magic time <" + target.split()[0] + (">: %.3f ms"
          % (np.median(evaluator(*datas).results) * 1000)))
    print(flush=True)
    print(flush=True)


def test_tvm(func, args):
    if target.startswith("llvm"):
        args[0] = 1
    log_file = func + "." + target.split()[0] + ".json"
    task = create_task(func, tuple(args))
    tune_task(task, log_file)
    time_task(task, log_file)
