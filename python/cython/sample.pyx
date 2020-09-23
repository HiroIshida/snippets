import cython
import numpy as np
cimport numpy as cnp

# pythonから呼び出す関数
def func1(int n):
    cdef:
        int i, sum
        list hoge
    sum = 0
    hoge = []
    for i in range(n):
        sum += i
        hoge.append(i)
    return sum, hoge


# pythonからは参照しない関数。cythonの中だけで使う場合
cdef func2(cnp.ndarray temp):
    """
    tempが１次元のndarrayとき
    """
    cdef:
        int i, N, sum
    sum = 0
    N = len(temp)
    for i in range(N):
        sum += temp[i]
    return sum

# pythonとcython両方から参照する場合で、cythonから参照するときに高速化したい場合
cpdef func3():
    pass
