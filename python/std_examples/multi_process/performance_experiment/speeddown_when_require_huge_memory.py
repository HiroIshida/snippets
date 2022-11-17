import numpy as np
import pickle
from dataclasses import dataclass
import time
import multiprocessing
import os
import subprocess
import numpy as np


def split_number(n_total, n_split):
    return [n_total // n_split + (1 if x < n_total % n_split else 0) for x in range(n_split)]


def task(args):
    n_iter, idx, matrix_size = args
    #cores = "{},{}".format(2 * idx, 2 * idx+1)
    #os.system("taskset -p -c {} {}".format(cores, os.getpid()))
    for _ in range(n_iter):
        A = np.random.randn(matrix_size, matrix_size)
        for _ in range(100):
            A = A.dot(A)


def measure_time(n_process: int, matrix_size: int) -> float:
    n_total = 100
    assigne_list = split_number(n_total, n_process)
    pool = multiprocessing.Pool(n_process)
    ts = time.time()
    pool.map(task, zip(assigne_list, range(n_process), [matrix_size] * n_process))
    elapsed = time.time() - ts
    return elapsed


if __name__ == "__main__":
    n_experiment_sample = 5
    n_logical = os.cpu_count()
    n_physical = int(0.5 * n_logical)
    result = {}
    for mat_size in [5, 10, 20, 40, 80, 160]:
        subresult = {}
        result[mat_size] = subresult
        for n_process in range(1, n_physical + 1):
            elapsed = np.mean([measure_time(n_process, mat_size) for _ in range(n_experiment_sample)])
            subresult[n_process] = elapsed
            print("{}, {}, {}".format(mat_size, n_process, elapsed))
    with open("result.pkl", "wb") as f:
        pickle.dump(result, f)
