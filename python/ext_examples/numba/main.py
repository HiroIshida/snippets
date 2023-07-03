import time
import numpy as np
from dataclasses import dataclass
import numba


def pure_compute(pts, b_min, b_max, N):
    ts = time.time()
    width = (b_max - b_min) / N

    min_height = 0.0
    M = np.ones((N, N)) * min_height

    for point in points:
        index = ((point[:2] - b_min) // width).astype(int)
        if np.any(index < 0):
            continue
        if np.any(index > N - 1):
            continue
        M[index[0], index[1]] = max(min(M[index[0], index[1]], point[2]), min_height)

    print("time: {}".format(time.time() - ts))


@numba.njit
def numba_compute(pts, b_min, b_max, N):
    width = (b_max - b_min) / N
    min_height = 0.0
    M = np.ones((N, N)) * min_height

    for point in pts:
        index = ((point[:2] - b_min) // width).astype(np.int32)
        if np.any(index < 0):
            continue
        if np.any(index > N - 1):
            continue
        M[index[0], index[1]] = max(min(M[index[0], index[1]], point[2]), min_height)
    return M


def test_numba_compute(points, b_min, b_max, N):
    ts = time.time()
    M = numba_compute(points, b_min, b_max, N)
    print("time: {}".format(time.time() - ts))


points = np.random.randn(10000, 3)
points2 = np.random.randn(10000, 3)
b_min = -2 * np.ones(2)
b_max = +2 * np.ones(2)
N = 100
pure_compute(points, b_min, b_max, N)
pure_compute(points2, b_min, b_max, N)
test_numba_compute(points, b_min, b_max, N)
test_numba_compute(points2, b_min, b_max, N)

# time: 0.14950227737426758
# time: 0.13654637336730957
# time: 1.1502599716186523
# time: 0.002474546432495117
