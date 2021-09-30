import dill
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

N = 50
b_min = np.zeros(3)
b_max = np.ones(3)
Ls = [np.linspace(l, h, N) for l, h in zip(b_min, b_max)]
mgrids = np.meshgrid(*Ls)
pts = np.array(zip(*[mg.flatten() for mg in mgrids]))

def func(X):
    X = X - np.atleast_2d(np.ones(3) * 0.5)
    a = 0.1
    b = 0.2
    c = 0.4
    X1 = X[:, 0]
    X2 = X[:, 1]
    X3 = X[:, 2]
    f = X1**2/(a**2) + (X2**2)/(b**2) + (X3**2)/(c**2) - 1.0
    return f

fs = func(pts)
print(fs)

spacing = (b_min - b_max)/(N-1)
F = fs.reshape(N, N, N)
F = np.swapaxes(F, 0, 1) # important!!!

verts, faces, _, _ = measure.marching_cubes_lewiner(F, 0, spacing=spacing)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2], alpha=0.1)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()
