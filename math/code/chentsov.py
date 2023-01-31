import numpy as np
import copy

def deconvert(xi):
    p = np.array([xi[0], xi[1], 1 - xi[0] - xi[1]])
    return p

def convert(p):
    xi = np.array([p[0], p[1]])
    return xi

def jacobian(xi):
    return np.array([[1, 0], [0, 1], [-1, -1]])

def gen_metric_tensor(jacobian): # matrix
    def metric_tensor(xi):
        jac = jacobian(xi)
        mat = np.zeros((2, 2))
        p = deconvert(xi)
        for i in range(3):
            grad = jac[i].reshape(1, 2)
            mat_ = (grad.T).dot(grad)
            mat += 1.0/p[i] * mat_
        return mat
    return metric_tensor

"""
def metric_derivative(xi, metric_tensor, i):
    p0 = deconvert(xi)
    A0 = metric_tensor(p0)

    xi_ = copy.copy(xi)
    eps = 1e-3
    xi_[i] += eps
    p_ = deconvert(xi_)
    return (xi_ - xi)/eps
"""

metric = gen_metric_tensor(jacobian)

# plot
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
fig = plt.figure()
ax = fig.add_subplot(111)
xi_1_seq = np.linspace(0, 1.0, 50)
xi_2_seq = np.linspace(0, 1.0, 50)
for xi_1 in xi_1_seq:
    for xi_2 in xi_2_seq:
        if xi_1 + xi_2 < 0.9:
            A = metric([xi_1, xi_2])
            try:
                vals, vecs = np.linalg.eig(A)
                if vals[0] < vals[1]:
                    val_large = vals[1]
                    val_small = vals[0]
                    vec_large = vecs[1]
                    vec_small = vecs[0]
                else:
                    val_large = vals[0]
                    val_small = vals[1]
                    vec_large = vecs[0]
                    vec_small = vecs[1]

                start = np.array([xi_1, xi_2])
                scale = 0.002
                end1 = start + vec_small * scale * val_small
                end2 = start - vec_small * scale * val_small
                end3 = start + vec_large * scale * val_large
                end4 = start - vec_large * scale * val_large
                for end in [end1, end2, end3, end4]:
                    line = mlines.Line2D([start[0], end[0]], [start[1], end[1]])
                    ax.add_line(line)
            except:
                pass






