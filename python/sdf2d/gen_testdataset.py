import numpy as np
import matplotlib.pyplot as plt 

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

n1 = n2 = 200
xlin = np.linspace(-1.0, 1.0, n1)
ylin = np.linspace(-1.0, 1.0, n2)
X, Y = np.meshgrid(xlin, ylin)

pts = np.array(zip(X.flatten(), Y.flatten()))
fs_ = sdf_combine(pts)
fs = fs_.reshape((n1, n2))

fig, ax = plt.subplots()
X, Y = np.meshgrid(xlin, ylin)
c = ax.contourf(X, Y, fs)
cbar = fig.colorbar(c)
plt.show()

