import numpy as np
from scipy.spatial import ConvexHull

import numpy as np
Z = np.random.randn(30, 2)

hull = ConvexHull(points=Z)
normals = hull.equations[:, :-1]
offset = hull.equations[:, -1]

# sample from bound
width = hull.max_bound - hull.min_bound
margin = width * 0.1
Z_sample = np.random.rand(1000, 2) * (width + margin * 2) + hull.min_bound - margin
z_inside_list = []

custom_offset = 0.0
for z in Z_sample:
    val = normals.dot(z) + offset
    if np.all(val < custom_offset):
        z_inside_list.append(z)
Z_sample_inside = np.array(z_inside_list)

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.scatter(Z[:, 0], Z[:, 1], color="b")
ax.scatter(Z_sample_inside[:, 0], Z_sample_inside[:, 1], color="r")
plt.show()
