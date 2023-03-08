import tqdm
import numpy as np
import matplotlib.pyplot as plt


def compute_geodesic(X: np.ndarray, r: float) -> np.ndarray:
    connection_list = []
    for i in tqdm.tqdm(range(len(X))):
        x = X[i]
        dists = np.sqrt(np.sum((X - x) ** 2, axis=1))
        indices_connected = np.where(dists < r)[0]
        connection_list.append(indices_connected)

    n = len(X)
    costs = np.ones(n) * np.inf
    costs[0] = 0
    prevs = np.zeros(n, dtype=int) - 1
    unvisited = list(range(n))

    for _ in tqdm.tqdm(range(n)):
        tmp_idx_min = np.argmin(costs[np.array(unvisited)])
        idx_u = unvisited.pop(tmp_idx_min)
        indices = connection_list[idx_u]
        for idx_v in indices:
            d = np.linalg.norm(X[idx_v] - X[idx_u])
            if costs[idx_v] > costs[idx_u] + d:
                costs[idx_v] = costs[idx_u] + d
                prevs[idx_v] = idx_u
    return costs

if __name__ == "__main__":
    n = 3000
    X = np.random.randn(n, 3)
    dists = np.sqrt(np.sum(X ** 2, axis=1))
    X = X / dists[:, None]

    r = 0.2
    costs = compute_geodesic(X, r)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=costs)
    plt.show()
