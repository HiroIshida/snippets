import os
import matplotlib.pyplot as plt
import time
from concurrent.futures import ProcessPoolExecutor

def generate_random(t_wait) -> int:
    time.sleep(t_wait)
    print("pid: {}, t_wait: {}".format(os.getpid(), t_wait))
    return t_wait

t_wait_list = [3.0] + [0] * 9

ts = time.time()
data = [(0, 0)]
c = 0
with ProcessPoolExecutor(2) as executor:
    for e in executor.map(generate_random, t_wait_list):
        elapsed = time.time() - ts
        c += 1
        data.append((elapsed, c))


ts, cs = zip(*data)
plt.plot(ts, cs, "x-", markersize=8)
plt.xlabel("time elapsed")
plt.ylabel("count")
plt.show()
