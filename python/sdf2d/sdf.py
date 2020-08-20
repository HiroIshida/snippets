import numpy as np
import matplotlib.pyplot as plt 
import scipy.spatial as spatial
from gen_testdataset import SampleTestData
import time
from math import *

std = SampleTestData(20, n_interval=3)
pts = np.array(zip(std.X.flatten(), std.Y.flatten()))

# E is list of index pair
V = std.c
n_vert = len(V)
E = np.array(zip(range(n_vert), np.array(range(n_vert)) + 1))
E[n_vert-1][1] = 0 

b_min = std.b_min
b_max = std.b_max
w = b_max - b_min
N = std.ns[0]

# first, do some scaling 
scale = N/w
V_reshaped = (V - np.repeat(b_min.reshape(1, 2), n_vert, 0)) \
        * np.repeat(scale.reshape(1, 2), n_vert, 0)

P_incrementer = []
for e in E:
    p, q = V_reshaped[e[0]], V_reshaped[e[1]]
    if p[1] > q[1]: # p is lower in y axis
        p, q = q, p
    dif = q - p
    inc = dif[1]/dif[0] # TODO assert inc != 0

    ymin, ymax = int(ceil(p[1])), int(floor(q[1])) # integer! 

    x_intersection = lambda y: (y - p[1])/inc + p[0]
    ceiled_x_intersection = lambda y: int(ceil(x_intersection(y))) # => integer
    for y in [ymin + i for i in range(ymax - ymin + 1)]: # for(int i=ymin; i<=y_max;  i++)
        x_isc = ceiled_x_intersection(y)
        P_incrementer.append((x_isc, y))

P_incrementer = np.array(P_incrementer)

fig, ax = plt.subplots()
ax.scatter(V_reshaped[:, 0], V_reshaped[:, 1], c="red")
ax.scatter(P_incrementer[:, 0], P_incrementer[:, 1])

ax.set_xticks(np.arange(0, N, 1))
ax.set_yticks(np.arange(0, N, 1))
plt.grid()
plt.show()
