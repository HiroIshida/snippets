# Kégl, Balázs. "Intrinsic dimension estimation using packing numbers." Advances in neural information processing systems 15 (2002).

import numpy as np
import matplotlib.pyplot as plt
import tqdm
from math import isnan, log
from skdim.datasets import hyperSphere
from typing import Optional

def pack_ball(X: np.ndarray, r: float):
    covered = np.zeros(len(X), dtype=bool)
    centers = []

    while True:
        non_covered = np.logical_not(covered)
        if sum(non_covered) == 0:
            break
        idx = np.where(non_covered)[0][0]
        c = X[idx]
        centers.append(c)
        covered_local = np.sum((X - c) ** 2, axis=1) - r ** 2 < 0
        covered = np.logical_or(covered, covered_local)
        ratio = sum(covered) / len(X)
    return centers


if __name__ == "__main__":
    dim_gt = 2
    n = 10000
    points = hyperSphere(n, dim_gt) + np.random.rand(n, dim_gt) * 0.3
    n_packing_list = []
    r_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for r in r_list:
        n_packing = 100000000
        for _ in tqdm.tqdm(range(20)):
            centers = pack_ball(points, r * 2)
            n_packing = min(len(centers), n_packing)
        n_packing_list.append(n_packing)

    arr = np.array(r_list) * 2
    X = np.log(arr[1:]) - np.log(arr[:-1])

    arr = np.array(n_packing_list)
    Y = np.log(arr[1:]) - np.log(arr[:-1])

    dims = -Y / X
    plt.plot(dims)
    plt.show()
