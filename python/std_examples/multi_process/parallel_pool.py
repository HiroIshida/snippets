# https://stackoverflow.com/questions/10415028/how-can-i-recover-the-return-value-of-a-function-passed-to-multiprocessing-proce
import multiprocessing
import time
import numpy as np
from multiprocessing import Process, Pool

lst = []

def long_task(procnum):
    return procnum**2


p = Pool(8)
result = p.map(long_task, [i for i in range(8)])
print(result)
