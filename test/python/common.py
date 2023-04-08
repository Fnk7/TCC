import os
import tvm
import numpy as np
from tvm import auto_scheduler

os.environ["TVM_NUM_THREADS"] = "72"

doTune = True
targets = ["llvm", "cuda", "opencl"]


def print_compute_dag(task):
    print("Computational DAG:")
    print(task.compute_dag, flush=True)


def tune_task(task, log_file):
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=2000,
        builder=auto_scheduler.LocalBuilder(timeout=100),
        runner=auto_scheduler.LocalRunner(timeout=100),
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
    )
    if doTune:
        task.tune(tune_option)


def time_task(task, log_file, target):
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
        func.entry_name, dev, repeat=1, min_repeat_ms=100)
    print("0v0 Magic time line <" + target + (">: %.3f ms"
          % (np.median(evaluator(*datas).results) * 1000)))
    print(flush=True)
    print(flush=True)
    datas.clear()
