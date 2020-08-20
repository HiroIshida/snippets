import numpy as np
import matplotlib.pyplot as plt 
from skimage import measure

def sdf_sphere(X, c, r):
    n_points = X.shape[0]
    C = np.repeat(c.reshape(1, 2), n_points, axis=0)
    dists = np.sqrt(np.sum((X - C)**2, 1))
    return dists - r

def sdf_combine(X):
    f1 = sdf_sphere(X, np.array([0, -0.4]), 0.6)
    f2 = sdf_sphere(X, np.array([0, 0.4]), 0.4)
    logicals = f1 > f2
    return f2 * logicals + f1 * (~logicals)

ns = np.array([200, 200])
b_min = np.array([-1.0, -1.0])
b_max = np.array([1.0, 1.0])
xlin, ylin = [np.linspace(b_min[i], b_max[i], ns[i]) for i in range(2)]
X, Y = np.meshgrid(xlin, ylin)

pts = np.array(zip(X.flatten(), Y.flatten()))
fs_ = sdf_combine(pts)
fs = fs_.reshape(ns)

fig, ax = plt.subplots()
X, Y = np.meshgrid(xlin, ylin)
c = ax.contourf(X, Y, fs)
cbar = fig.colorbar(c)

c__ = measure.find_contours(fs.T, 0.0)[0] # only one closed curve now
def rescale_contour(pts, b_min, b_max, n):
    n_points, n_dim = pts.shape
    width = b_max - b_min
    b_min_tile = np.tile(b_min, (n_points, 1))
    width_tile = np.tile(width, (n_points, 1))
    pts_rescaled = b_min_tile + width_tile * pts / (n - 1)
    return pts_rescaled

c_ = c__[::10]
c = rescale_contour(c_, b_min, b_max, ns[0])

ax.scatter(c[:, 0], c[:, 1])
plt.show()


