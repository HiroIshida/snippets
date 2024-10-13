import time
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import plotly.graph_objects as go


@njit
def compute_distance_field2(binary_map):
    max_val = 1024  # for now
    N, M, L = binary_map.shape
    dmap = np.empty((N, M, L), dtype=np.uint16)

    for i in range(N):
        for j in range(M):
            for k in range(L):
                if binary_map[i, j, k]:
                    dmap[i, j, k] = 0
                else:
                    dmap[i, j, k] = max_val

    for i in range(N):
        for j in range(M):
            for k in range(L):
                if dmap[i, j, k] > 0:
                    val_previous_k = max_val
                    if k > 0:
                        val_previous_k = dmap[i, j, k - 1] + 1

                    val_previous_j = max_val
                    if j > 0:
                        val_previous_j = dmap[i, j - 1, k] + 1

                    val_previous_i = max_val
                    if i > 0:
                        val_previous_i = dmap[i - 1, j, k] + 1

                    val_previous = min(min(val_previous_j, val_previous_i), val_previous_k)
                    if val_previous < dmap[i, j, k]:
                        dmap[i, j, k] = val_previous

    for i in range(N - 1, -1, -1):
        for j in range(M - 1, -1, -1):
            for k in range(L - 1, -1, -1):
                if dmap[i, j, k] > 0:
                    val_next_k = max_val
                    if k < L - 1:
                        val_next_k = dmap[i, j, k + 1] + 1

                    val_next_j = max_val
                    if j < M - 1:
                        val_next_j = dmap[i, j + 1, k] + 1

                    val_next_i = max_val
                    if i < N - 1:
                        val_next_i = dmap[i + 1, j, k] + 1

                    val_next = min(min(val_next_j, val_next_i), val_next_k)
                    if val_next < dmap[i, j, k]:
                        dmap[i, j, k] = val_next
    return dmap

def visualize_3d_voxels(dmap, threshold):
    mask = dmap <= threshold
    x, y, z = np.where(mask)
    fig = go.Figure(data=go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=5,
            color=dmap[mask],
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(title="Distance"),
        )
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        title="3D Voxel Visualization of Distance Field"
    )
    fig.show()


if __name__ == "__main__":
    N = 100
    M = 120
    L = 80
    binary_map = np.zeros((N, M, L))
    binary_map[10:40, 10:30, 10:30] = 1
    binary_map[60:80, 50:80, 20:60] = 1
    binary_map = binary_map.astype(bool)
    dmap = compute_distance_field2(binary_map)

    ts = time.time()
    for _ in range(100):
        dmap = compute_distance_field2(binary_map)
    print(f"elapsed per iteration: {(time.time() - ts) / 100:.6f}")
    print(dmap.shape)
    visualize_3d_voxels(dmap, 10)
