import numpy as np
import matplotlib.pyplot as plt 
import scipy.spatial as spatial
from gen_testdataset import SampleTestData
import time
from math import *

std = SampleTestData(30, n_interval=3)
pts = np.array(zip(std.X.flatten(), std.Y.flatten()))

b_min = std.b_min
b_max = std.b_max
w = b_max - b_min
N = std.ns[0]
V = std.V
E = std.E
debug_view = True


ts = time.time()
# first, do some scaling 

n_vert = len(V)
scale = (N-1)/w
V_rescaled = (V - np.repeat(b_min.reshape(1, 2), n_vert, 0)) \
        * np.repeat(scale.reshape(1, 2), n_vert, 0)

scale = (N-1)/w # TODO  N-1 ?? 
pts_scaled = (pts - np.repeat(b_min.reshape(1, 2), N*N, 0)) \
        * np.repeat(scale.reshape(1, 2), N*N, 0)

# TODO x based index
P_incrementer = []
incrementer_map = [[] for i in range(N)]
for e in E:
    p, q = V_rescaled[e[0]], V_rescaled[e[1]]
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
        incrementer_map[y].append(x_isc)

P_incrementer = np.array(P_incrementer)

intersection_count = np.zeros((N, N))
for j in range(N):
    intersection_count[0][j] = 0
    for i in [ii+1 for ii in range(N-1)]: # for(int i=1; i<N; i++)
        intersection_count[i][j] = intersection_count[i-1][j]  
        if i in incrementer_map[j]:
            intersection_count[i][j] += 1

idxes_inside = intersection_count.reshape(N*N) % 2 == 1

if debug_view:
    fig, ax = plt.subplots()
    ax.scatter(V_rescaled[:, 0], V_rescaled[:, 1], c="red")
    ax.scatter(P_incrementer[:, 0], P_incrementer[:, 1])
    ax.set_xticks(np.arange(0, N, 1)); ax.set_yticks(np.arange(0, N, 1))

    idxes_inside = intersection_count.T.reshape(N*N) % 2 == 1
    ax.scatter(pts_scaled[idxes_inside, 0], pts_scaled[idxes_inside, 1], c="green", marker="x")

    plt.grid(); plt.show()

print(time.time() - ts)
