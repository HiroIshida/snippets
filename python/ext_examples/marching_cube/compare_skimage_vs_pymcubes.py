import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure
import mcubes

N = 20
w = 1.2
x = y = z = np.linspace(-w, w, N) 
X, Y, Z = np.meshgrid(x, y, z)
pts = np.array(zip(X.flatten(), Y.flatten(), Z.flatten()))
def func(X):
    r_lst = np.sum(X ** 2, axis=1)
    return np.cos(r_lst * 3.14)

F = func(pts).reshape(N, N, N)

import time
ts = time.time()
verts, faces = measure.marching_cubes(F, 0.0)
print(time.time() - ts)
ts = time.time()
verts, faces = mcubes.marching_cubes(F, 0)
print(time.time() - ts)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2])
plt.show()
