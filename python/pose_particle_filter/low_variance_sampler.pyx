import cython
import numpy as np
cimport numpy as cnp
from libc.stdlib cimport rand, RAND_MAX

def low_variance_sampler_cython(cnp.ndarray[double, ndim=1] ws, double r_real):
    # ws is numpya array
    cdef int N = ws.shape[0]
    cdef cnp.ndarray[int, ndim=1] idxes
    idxes = np.empty(N, dtype=np.int32)

    cdef double w_sum = 0.0
    for i in range(N):
        w_sum += ws[i]

    cdef double r = r_real * (1.0/N)
    cdef double c = ws[0]/w_sum
    cdef int k = 0
    cdef double U
    for n in range(N):
        U = r + n*(1.0/N)
        while U > c:
            k+=1
            c = c+ws[k]/w_sum
        idxes[n] = k
    return idxes
