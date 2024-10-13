import time
import numpy as np
import matplotlib.pyplot as plt
from numba import njit


def compute_distance_field(binary_map: np.ndarray) -> np.ndarray:
    max_val = 1024  # for now
    N, M = binary_map.shape
    dmap = np.ones((N, M), dtype=int) * max_val
    dmap[binary_map == 1] = 0
    for i in range(N):
        for j in range(M):
            val_previous_j = max_val if j == 0 else dmap[i, j-1] + 1
            val_previous_i = max_val if i == 0 else dmap[i-1, j] + 1
            val_previous = min(val_previous_j, val_previous_i)
            dmap[i, j] = min(dmap[i, j], val_previous)

    for i in range(N-1, -1, -1):
        for j in range(M-1, -1, -1):
            val_previous_j = max_val if j == M-1 else dmap[i, j+1] + 1
            val_previous_i = max_val if i == N-1 else dmap[i+1, j] + 1
            val_previous = min(val_previous_j, val_previous_i)
            dmap[i, j] = min(dmap[i, j], val_previous)
    return dmap


@njit
def compute_distance_field2(binary_map):
    max_val = 1024  # for now
    N, M = binary_map.shape
    dmap = np.empty((N, M), dtype=np.uint16)

    for i in range(N):
        for j in range(M):
            if binary_map[i, j]:
                dmap[i, j] = 0
            else:
                dmap[i, j] = max_val

    for i in range(N):
        for j in range(M):
            if dmap[i, j] > 0:
                val_previous_j = max_val
                if j > 0:
                    val_previous_j = dmap[i, j - 1] + 1

                val_previous_i = max_val
                if i > 0:
                    val_previous_i = dmap[i - 1, j] + 1

                val_previous = min(val_previous_j, val_previous_i)
                if val_previous < dmap[i, j]:
                    dmap[i, j] = val_previous

    for i in range(N - 1, -1, -1):
        for j in range(M - 1, -1, -1):
            if dmap[i, j] > 0:
                val_previous_j = max_val
                if j < M - 1:
                    val_previous_j = dmap[i, j + 1] + 1

                val_previous_i = max_val
                if i < N - 1:
                    val_previous_i = dmap[i + 1, j] + 1

                val_previous = min(val_previous_j, val_previous_i)
                if val_previous < dmap[i, j]:
                    dmap[i, j] = val_previous
    return dmap

if __name__ == "__main__":
    N = 100
    M = 120
    binary_map = np.zeros((N, M))
    binary_map[10:40, 10:30] = 1
    binary_map[60:80, 50:80] = 1
    binary_map = binary_map.astype(bool)
    dmap = compute_distance_field2(binary_map)

    ts = time.time()
    for _ in range(100):
        dmap = compute_distance_field(binary_map)
    print(f"elapsed per iteration: {(time.time() - ts) / 100:.6f}")

    ts = time.time()
    for _ in range(100):
        dmap = compute_distance_field2(binary_map)
    print(f"elapsed per iteration: {(time.time() - ts) / 100:.6f}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(binary_map, cmap='gray')
    ax[0].set_title('Binary Pixel Map')
    ax[0].axis('off')

    im = ax[1].imshow(dmap, cmap='viridis')
    ax[1].set_title('Chamfer Distance Field (3-4 distance)')
    fig.colorbar(im, ax=ax[1])
    plt.show()
