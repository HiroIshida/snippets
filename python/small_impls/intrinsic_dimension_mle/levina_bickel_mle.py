import numpy as np
import tqdm
from math import isnan, log
from skdim.datasets import hyperSphere
from typing import Optional

# implementation of Levina-Bicket algorithm
# Levina, Elizaveta, and Peter Bickel. "Maximum likelihood estimation of intrinsic dimension." Advances in neural information processing systems 17 (2004).


def determine_id(center: np.ndarray, points: np.ndarray, radius: float) -> Optional[int]:
    dists = np.sqrt(np.sum((center - points)**2, axis=1))
    indices = np.argsort(dists)
    dists_sorted = dists[indices]
    dists_sorted_inside = dists_sorted[np.logical_and(dists_sorted < radius, dists_sorted > 0)]

    n_inside = len(dists_sorted_inside)
    if n_inside == 0:
        return None
    dim = (np.sum(np.log(radius / dists_sorted_inside)) / n_inside)**(-1)
    return dim


def determine_average_id(points: np.ndarray, radius: float, n_neighbour: int = 100) -> float:
    representative_indices = np.random.choice(list(range(n_neighbour)), size=n_neighbour, replace=False)
    dims = []
    for ind in tqdm.tqdm(representative_indices):
        dim = determine_id(points[ind], points, radius)
        if dim is None:
            continue
        dims.append(dim)
    return np.mean(dims)


if __name__ == "__main__":
    dim_gt = 4
    n = 10000
    points = hyperSphere(n, dim_gt) + np.random.randn(n, dim_gt) * 0.3
    dim = determine_average_id(points, 1.0)
    print(dim)
