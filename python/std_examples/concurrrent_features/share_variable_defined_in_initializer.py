import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime


def init() -> None:
    unique_seed = datetime.now().microsecond + os.getpid()
    global shared
    shared = unique_seed


def gen(_) -> int:
    global shared
    return shared


n_cpu = 24
with ProcessPoolExecutor(n_cpu, initializer=init) as executor:
    mapped = executor.map(gen, range(100))
n_unique = len(set(list(mapped)))
assert n_unique == n_cpu
