import copy
import numpy as np

def gen(n_dim: int, n_split: int) -> np.ndarray:
    assert n_dim > 0
    arr = np.expand_dims(np.linspace(0, 1, n_split), axis=0).T
    for i in range(n_dim-1):
        row, col = arr.shape
        partial_list = []
        for val in np.linspace(0, 1, n_split):
            partial = np.ones((row, col + 1)) * val
            partial[:, 1:] = arr
            partial_list.append(partial)
        arr = np.vstack(partial_list)
    return arr


if __name__ == "__main__":
    arr = gen(3, 3)
    print(len(arr))
    print(arr)
