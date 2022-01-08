import multiprocessing
import time
import numpy as np
from multiprocessing import Process, Queue, Lock, Array

def long_task(l, i, lst):
    with l:
        lst[i] = i

result = Array('i', 10)
procs = []
lock = Lock()
for i in range(10):
    proc = Process(target=long_task, args=[lock, i, result])
    proc.start()
    procs.append(proc)
for proc in procs:
    proc.join()

print(result[:])
