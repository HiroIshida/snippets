import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri
from skimage import measure
import numpy as np

# data generation
N = 6
w = 0.8
xlin, ylin, zlin = [np.linspace(-w, w, N) for i in [0, 1, 2]]
X, Y, Z = np.meshgrid(xlin, ylin, zlin)
pts = np.array([[x, y, z] for (x, y, z) in zip(X.flatten(), Y.flatten(), Z.flatten())])
fun = lambda x: np.sqrt(x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**2) - 0.5
fs = fun(pts).reshape([N]*3)
V, F, _, _ = measure.marching_cubes(fs, 0.0)

def make_tripod(three_verts, shift=0):
    four_verts = np.vstack((three_verts, [0, 0, 0]))
    print(four_verts)
    triangles = np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]]) + shift
    return four_verts, triangles

mytri = lambda V, F, ax: ax.plot_trisurf(V[:, 0], V[:, 1], V[:, 2], triangles=F, cmap=plt.cm.Spectral, alpha=0.5)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(V[:, 0], V[:, 1], V[:, 2], c="r", s=1)

green = False
if green:
    for f in F:
        V_new, F_new = make_tripod(V[f])
        mytri(V_new, F_new, ax)
else:
    mytri(V, F, ax)
plt.show()


