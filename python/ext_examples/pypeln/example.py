import numpy as np
import pypeln as pl
import time

t_sleep = 0.1

def f(x):
    time.sleep(t_sleep)
    return x + 1

def g(x):
    time.sleep(t_sleep)
    return x ** 2


X = list(np.random.randint(100, size=100))

# serial
ts = time.time()
ret_list = []
for x in X:
    ret = g(f(x))
    ret_list.append(ret)
print(f"Serial: {time.time() - ts:.2f} s")

# pipeline
ts = time.time()
stage = pl.process.map(f, X, workers=1, maxsize=1)
stage = pl.process.map(g, stage, workers=1, maxsize=1)
ret_list_pp = list(stage)
print(f"Pipeline: {time.time() - ts:.2f} s")

assert ret_list == ret_list_pp, f"ret_list: {ret_list}, ret_list_pp: {ret_list_pp}"
