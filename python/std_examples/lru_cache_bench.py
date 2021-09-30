import contextlib
from datetime import datetime
import functools

if hasattr(functools, 'lru_cache'):
    _functools_lru_cache = functools.lru_cache
else:
    try:
        import functools32
    except ImportError:
        functools.lru_cache  # install `functools32` to run on python2.7
    else:
        _functools_lru_cache = functools32.lru_cache

from repoze.lru import lru_cache
import pylru
import numpy as np


@contextlib.contextmanager
def measure(name, log_enable=True):
    import time
    s = time.time()
    yield
    e = time.time()
    if log_enable:
        print("{}: {:.4f}sec".format(name, e - s))


@lru_cache(maxsize=1000)
def random_generate_with_repoze(max_number):
    return np.random.randint(max_number)


@_functools_lru_cache(maxsize=1000)
def random_generate_with_functools(max_number):
    return np.random.randint(max_number)


@pylru.lrudecorator(1000)
def random_generate_with_pylru(max_number):
    return np.random.randint(max_number)


N = 100000
max_number = 100000

np.random.seed(0)
with measure('repoze'):
    for i in range(1, N):
        random_generate_with_repoze(i)

np.random.seed(0)
with measure('functools'):
    for i in range(1, N):
        random_generate_with_functools(i)

np.random.seed(0)
with measure('pylru'):
    for i in range(1, N):
        random_generate_with_pylru(i)

np.random.seed(0)
with measure('repoze'):
    for _ in range(1, N):
        value = np.random.randint(max_number) + 1
        random_generate_with_repoze(value)

np.random.seed(0)
with measure('functools'):
    for _ in range(1, N):
        value = np.random.randint(max_number) + 1
        random_generate_with_functools(value)

np.random.seed(0)
with measure('pylru'):
    for _ in range(1, N):
        value = np.random.randint(max_number) + 1
        random_generate_with_pylru(value)
