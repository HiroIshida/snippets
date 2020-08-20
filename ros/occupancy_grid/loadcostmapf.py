import scipy.interpolate  
import copy
import numpy as np
import matplotlib.pyplot as plt 
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
fig, ax = plt.subplots()
c = ax.contourf(X, Y, Z)
cbar = fig.colorbar(c)

idxes_clear = Z_ < 80
pts_valid = pts[idxes_clear, :]
#ax.scatter(pts_valid[:, 0], pts_valid[:, 1])
cs_unscaled = measure.find_contours(Z.T, 95.0)

def rescale_contour(pts, b_min, b_max, n):
    n_points, n_dim = pts.shape
    width = b_max - b_min
    b_min_tile = np.tile(b_min, (n_points, 1))
    width_tile = np.tile(width, (n_points, 1))
    pts_rescaled = b_min_tile + width_tile * pts / (n - 1)
    return pts_rescaled

cs = [rescale_contour(c, np.array([-b, -b]), np.array([b, b]), 200) for c in cs_unscaled]

for c in cs:
    ax.scatter(c[:, 0], c[:, 1])
plt.show() 
