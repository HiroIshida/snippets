import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import time
from datetime import datetime


def init() -> None:
    unique_seed = datetime.now().microsecond + os.getpid()
    np.random.seed(unique_seed)
    print("pid {}: random seed is set to {}".format(os.getpid(), unique_seed))


def generate_random(_) -> int:
    N_RAND = 100000000  # big number
    return np.random.randint(N_RAND)


n_cpu = 24

# note that we must change random seed per different process to
# generate non-duplicated random variables
rands = []
for _ in range(10):
    with ProcessPoolExecutor(n_cpu, initializer=init) as executor:
        mapped = executor.map(generate_random, range(100))
    rands.extend(list(mapped))

print("unique element: {}".format(len(set(rands))))
