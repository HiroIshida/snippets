import skrobot
import numpy as np
from skrobot.models import Box, MeshLink, Axis
from sklearn.covariance import EmpiricalCovariance
m = MeshLink("./forearm.obj")

debug = False
if debug:
    viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))
    viewer.add(m)
    viewer.show()

vertices_ = m.visual_mesh.vertices  
mean = np.mean(vertices_, axis=0)
vertices = np.array([v - mean for v in vertices_])
#vertices = vertices_ - mean[None, :]
cov = EmpiricalCovariance().fit(vertices)
W, V = np.linalg.eig(cov.covariance_)
pts_converted = vertices.dot(V)
radius_sqrt = pts_converted[:, 1]**2 + pts_converted[:, 2]**2
R = np.sqrt(np.max(radius_sqrt)) * 1.0

def condition_positive(h):
    sphere_heights = h + np.sqrt(R**2 - radius_sqrt)
    return np.all(sphere_heights > pts_converted[:, 0])

def condition_negative(h):
    sphere_heights = h - np.sqrt(R**2 - radius_sqrt)
    return np.all(sphere_heights < pts_converted[:, 0])

h_vert_max = np.max(pts_converted[:, 0])
h_vert_min = np.min(pts_converted[:, 0])
h_max_cands = np.linspace(0.0, h_vert_max, 50)
h_min_cands = np.linspace(0, h_vert_min, 50)

idx_p = np.where([condition_positive(h) for h in h_max_cands])[0][0]
idx_n = np.where([condition_negative(h) for h in h_min_cands])[0][0]
# max and min height of the swept sphere
h_max = h_max_cands[idx_p]
h_min = h_min_cands[idx_n]

import matplotlib.pyplot as plt 
fig, ax = plt.subplots()
ax.scatter(pts_converted[:, 1], pts_converted[:, 0])
c1 = plt.Circle((0, h_max), R, color='r', fill=False, ls="--", lw=2)
c2 = plt.Circle((0, h_min), R, color='r', fill=False, ls="--", lw=2)
ax.add_artist(c1)
ax.add_artist(c2)
ax.axis('equal')
plt.show()
