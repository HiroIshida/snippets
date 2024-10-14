import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def update(mat: np.ndarray, mat_new: np.ndarray) -> None:
    N, M = mat.shape
    for i in range(N):
        for j in range(M):
            neighbours = [(i-1, j-1), (i-1, j), (i-1, j+1), (i, j-1), (i, j+1), (i+1, j-1), (i+1, j), (i+1, j+1)]
            c = sum(mat[x % N, y % M] for x, y in neighbours)
            if mat[i, j]:
                if c < 2 or c > 3:
                    mat_new[i, j] = 0
                else:
                    mat_new[i, j] = 1
            else:
                if c == 3:
                    mat_new[i, j] = 1
                else:
                    mat_new[i, j] = 0

N = 56
mat = np.zeros((N, N), dtype=bool)
mat[20:30, 20:30] = np.random.choice([0, 1], (10, 10), p=[0.5, 0.5])
mat[40:50, 40:50] = np.random.choice([0, 1], (10, 10), p=[0.5, 0.5])

ts = time.time()
mat_3d = np.zeros((N, N, N), dtype=bool)
for t in range(N):
    mat_3d[:,:,t] = mat
    mat_new = mat.copy()
    update(mat, mat_new)
    mat = mat_new
print("Time: ", time.time() - ts)

# plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.voxels(mat_3d, edgecolor='k')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Time Step')
ax.set_title("Conway's Game of Life - 3D Visualization")
ax.view_init(elev=20, azim=45)
plt.show()
