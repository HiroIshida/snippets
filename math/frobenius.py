import numpy as np
import matplotlib.pyplot as plt
"""
test how the frobenius norm works between symmetric matrices
"""

x_lin, y_lin = [np.linspace(-2, 2, 100)]*2
X, Y = np.meshgrid(x_lin, y_lin)
H = np.diag([1, 2])

def calculate_dim(param):
    L = len(param)
    return int((-1 + np.sqrt(1 + 8*L)) * 0.5)

def generate_pair(n_dim):
    # this defines parameterization of the matrix

    for i in range(n_dim):
        # first fill the diagonal elements
        yield (i, i)

    for i in range(n_dim):
        for j in range(0, i, 1):
            yield (i, j)

def frobenius_norm(param):
    dim_matrix = calculate_dim(param)
    diag_norm = np.sum(param[:dim_matrix] ** 2)
    nondiag_norm = np.sum(param[dim_matrix:] ** 2) * 2 # because symmetric
    norm = np.sqrt(diag_norm + nondiag_norm)
    return norm

import time 
param = np.array([1, 1.0, 0.5])
norm = frobenius_norm(param)

V, E = np.linalg.eig([[1, 0.5], [0.5, 1.0]])
print(norm)
