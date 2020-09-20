from math import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
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

def frobenius_metric(param_dim):
    mat_dim = int((-1 + np.sqrt(1 + 8*param_dim)) * 0.5)

    M = np.eye(param_dim)
    for i in range(param_dim):
        M[i, i] = 2.0
    for i in range(mat_dim):
        M[i, i] = 1.0
    return M

def plot_ellipse(param, ax, color="blue", alpha=0.1):

    def symmetrize(A):
        return A + A.T - np.diag(A.diagonal())

    dim_mat = calculate_dim(param)
    M_tmp = np.eye(2)
    for ((i, j), val) in zip(generate_pair(dim_mat), param):
        M_tmp[i, j] = val
    M = symmetrize(M_tmp)
    E, V = np.linalg.eig(M)

    v_principle = V[:, 0]
    angle = atan2(v_principle[1], v_principle[0]) * 180 /pi

    e = Ellipse((0, 0), E[0], E[1], angle, color=color, fill=False)
    e.set_alpha(alpha)
    e.set_clip_box(ax.bbox)
    ax.add_artist(e)

import time 
param_center = np.array([1, 1.0, 0.5])

M = frobenius_metric(3)
params_rand = np.random.multivariate_normal(param_center, np.linalg.inv(M) * 0.05, 1000)

fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
b = 1.5
ax.set_xlim([-b, b])
ax.set_ylim([-b, b])
for param in params_rand:
    plot_ellipse(param, ax)
plot_ellipse(param_center, ax, color="red", alpha=1.0)
plt.show()
