import scipy.interpolate  
import copy
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as pat 
import pickle 
from skimage import measure

with open("costmapf.pickle", "rb") as f:
    costmapdata = pickle.load(f)

costmapf = costmapdata.convert2sdf()

b = 4.0
xlin = np.linspace(-b, b, 200)
ylin = np.linspace(-b, b, 200)
X, Y = np.meshgrid(xlin, ylin)
pts = np.array(list(zip(X.flatten(), Y.flatten())))
Z_ = costmapf(pts)
Z = Z_.reshape((200, 200))

idxes_clear = Z_ < 80
pts_valid = pts[idxes_clear, :]
#ax.scatter(pts_valid[:, 0], pts_valid[:, 1])

import time
ts = time.time()
cs_unscaled = measure.find_contours(Z.T, 95.0)

def rescale_contour(pts, b_min, b_max, n):
    n_points, n_dim = pts.shape
    width = b_max - b_min
    b_min_tile = np.tile(b_min, (n_points, 1))
    width_tile = np.tile(width, (n_points, 1))
    pts_rescaled = b_min_tile + width_tile * pts / (n - 1)
    return pts_rescaled

V_list = [rescale_contour(c[::5], np.array([-b, -b]), np.array([b, b]), 200) for c in cs_unscaled]

def make_edges(V):
    n_vert = len(V)
    E = np.array(zip(range(n_vert), np.array(range(n_vert)) + 1))
    E[n_vert-1][1] = 0 
    return E
E_list = map(make_edges, V_list)

V = V_list[0]; E = E_list[0]
for i in range(len(E_list)-1):
    E_append = E_list[i+1] + len(V)
    E = np.vstack((E, E_append))
    V_append = V_list[i+1]
    V = np.vstack((V, V_append))

import sdf2d
#V += np.random.randn(V.shape[0], V.shape[1]) * 1
sdf_ = sdf2d.convert2sdf(V, E, np.array([-b, -b]), np.array([b, b]), [200, 200])
sdf = np.array(sdf_)

print(time.time() - ts)

fig, ax = plt.subplots()
#c = ax.contourf(X, Y, Z)
c = ax.contourf(X, Y, sdf.T)
cbar = fig.colorbar(c)
for c in V_list:
    ax.scatter(c[:, 0], c[:, 1])
for e in E:
    v0 = V[e[0]]
    v1 = V[e[1]]
    ax.plot([v0[0], v1[0]], [v0[1], v1[1]], 'g')

for x in xlin:
    ax.plot([x, x], [-b, b])

for y in ylin:
    ax.plot([-b, b], [y, y])


plt.show() 

