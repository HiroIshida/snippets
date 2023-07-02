import time
import py_fast_marching_method as fmm  # https://github.com/HiroIshida/fast-marching-method
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


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
height_map[5:15, 5:15] = 8
height_map[30:40, 5:15] = 15
height_map[5:40, 25:35] = 20

ts = time.time()
voxel_grids = compute_occgrid(height_map, 28)
indices = np.where(voxel_grids)
indices = np.vstack(indices).T
arrival_times = fmm.uniform_speed_signed_arrival_time(np.array(size), indices, np.zeros(len(indices)), np.ones(3), 1.0)
print(time.time() - ts)

lins = [np.arange(size[i]) for i in range(3)]
X, Y, Z = np.meshgrid(*lins)

print(X.shape)
print(arrival_times.shape)
plt.hist(arrival_times.flatten())
plt.show()

fig = go.Figure(
    data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=arrival_times.flatten(),
        isomin=0.0,
        isomax=30,
        opacity=0.1, # needs to be small to see through all surfaces
        surface_count=20, # needs to be a large number for good volume rendering
        colorscale="jet_r",
    )
)
fig.show()
