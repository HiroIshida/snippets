import numpy as np
import matplotlib.pyplot as plt 
import scipy.spatial as spatial
from gen_testdataset import SampleTestData
import time

std = SampleTestData(n_interval=5)
std.show()
tree = spatial.KDTree(std.c)

pts = np.array(zip(std.X.flatten(), std.Y.flatten()))

ts = time.time()
tree.query(pts)
print(time.time() - ts)

ts = time.time()
n_pts = len(pts)
dists_list = []
for vert in std.c:
    Vert = np.repeat(vert.reshape(1, 2), n_pts, axis=0)
    dists = np.sum(np.sqrt((pts - Vert)**2), 1)
    dists_list.append(dists)
dists_list = np.array(dists_list)
mindists = np.min(dists_list, axis=0)
print(time.time() - ts)

std.show(mindists.reshape(std.ns))


