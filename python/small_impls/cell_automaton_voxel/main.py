import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numba import njit

@njit
def update(mat: np.ndarray, mat_new: np.ndarray) -> None:
    N, M = mat.shape
    for i in range(N):
        for j in range(M):
            neighbours = [(i-1, j-1), (i-1, j), (i-1, j+1), (i, j-1), (i, j+1), (i+1, j-1), (i+1, j), (i+1, j+1)]
            c = 0
            for nei in neighbours:
                x, y = nei
                c += mat[x%N, y%M]
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


def create_conway(init_layer: np.ndarray, height: int):
    mat_3d = np.zeros((init_layer.shape[0], init_layer.shape[1], height), dtype=bool)
    t = 0
    mat_3d[:, :, t] = init_layer
    for t in range(1, height):
        dx = np.random.randint(-1, 2)
        dy = np.random.randint(-1, 2)
        mat_shifted = np.roll(mat_3d[:, :, t-1], shift=(dx, dy), axis=(0, 1))
        mat_new = np.zeros_like(init_layer, dtype=bool)
        update(mat_shifted, mat_new)
        mat_3d[:, :, t] = mat_new
    return mat_3d


N = 56
mat = np.zeros((N, N), dtype=bool)
mat[20:30, 20:30] = np.random.choice([0, 1], (10, 10), p=[0.5, 0.5])
mat_3d = create_conway(mat, N)
mat = np.zeros((N, N), dtype=bool)
mat[30:40, 20:30] = np.random.choice([0, 1], (10, 10), p=[0.5, 0.5])
ts = time.time()
mat_3d = create_conway(mat, N)
print(f"Time: {time.time() - ts:.6f}")

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.voxels(mat_3d, edgecolor='k')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Time Step')
ax.set_title("Conway's Game of Life - 3D Visualization")
ax.view_init(elev=20, azim=45)
plt.show()
