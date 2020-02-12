# only python3
from functools import lru_cache
import time
import numpy as np

def example1():
    @lru_cache(maxsize = None)
    def heavy_process(x):
        print("too heavy to compute")
        time.sleep(2)
        return x**2

    heavy_process(0)
    heavy_process(0)

def examle2():
    '''
    you have function f(x) which produces multiple data at once, say data1 and data2.
    you want to use two of them at different timings.
    you can write two functions 
    f1(x) = lambda x: f(x)[0]
    f2(x) = lamdba x: f(x)[1]
    '''
    @lru_cache(maxsize = None)
    def data_gen(x):
        print("too heavy to compute")
        time.sleep(1)
        data1 = np.random.randn(10000) * x
        data2 = np.random.randn(10000) * -x
        return data1, data2

    def f_use_data1(x):
        data1, _ = data_gen(x)
        return np.sum(data1)

    def f_use_data2(x):
        data2, _ = data_gen(x)
        return np.sum(data2)

    print(f_use_data1(1))
    print(f_use_data2(1))










