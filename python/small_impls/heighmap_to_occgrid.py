import time
import matplotlib.pyplot as plt
import numpy as np

def compute_occgrid(height_map: np.ndarray, n_height_size: int):
    hm_shape = list(height_map.shape)
    shape = hm_shape + [n_height_size]
    voxel_grids = np.zeros(shape, dtype=bool)
    for i in range(hm_shape[0]):
        for j in range(hm_shape[1]):
            h = height_map[i, j]
            voxel_grids[i, j, :h] = True
    return voxel_grids


size = (56, 56, 28)
height_map = np.ones([56, 56], dtype=int)
height_map[3:15, 3:6] = 10
height_map[20:40, 10:30] = 20

ts = time.time()
voxel_grids = compute_occgrid(height_map, 28)
indices = np.where(voxel_grids)
print(time.time() - ts)
print(indices)

# ax = plt.figure().add_subplot(projection='3d')
# ax.voxels(voxel_grids, edgecolor='k')
# plt.show()
