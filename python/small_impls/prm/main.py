from dataclasses import dataclass
import numpy as np
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

class World:
    lb: np.ndarray
    radius: float = 0.12
    def __init__(self):
        self.lb = np.array([0.0, 0.0])
        self.ub = np.array([1.0, 1.0])

        np.random.seed(2)
        centers = []
        while True:
            center = np.random.rand(2)
            if np.linalg.norm(center - np.ones(2) * 0.1) < self.radius:
                continue
            if np.linalg.norm(center - np.array([0.9, 0.1])) < self.radius:
                continue
            centers.append(center)
            if len(centers) == 12:
                break
        self.centers = np.array(centers)

    def is_valid(self, x: np.ndarray) -> bool:
        return self.is_valid_batch(x.reshape(1, -1))[0]

    def is_valid_batch(self, X: np.ndarray) -> np.array:
        assert X.shape[1] == 2
        valid = np.ones(X.shape[0], dtype=bool)
        for center in self.centers:
            valid = np.logical_and(valid, np.linalg.norm(X - center, axis=1) > self.radius)
        valid = np.logical_and(valid, np.all(X >= self.lb, axis=1))
        valid = np.logical_and(valid, np.all(X <= self.ub, axis=1))
        return valid

    def visualize(self) -> Tuple:
        fig, ax = plt.subplots()
        ax.set_xlim(self.lb[0] - 0.1, self.ub[0] + 0.1)
        ax.set_ylim(self.lb[1] - 0.1, self.ub[1] + 0.1)
        ax.set_aspect('equal')
        square_x = [0, 1, 1, 0, 0]
        square_y = [0, 0, 1, 1, 0]
        ax.plot(square_x, square_y, 'k-')
        for center in self.centers:
            circle = Circle(center, self.radius, color='r', fill=True)
            ax.add_artist(circle)
        return fig, ax


def build_prm_graph(world: World, start: np.ndarray, goal: np.ndarray, num_nodes: int, radius: float = 0.1) -> Tuple:
    assert world.is_valid(start)
    assert world.is_valid(goal)
    X_cand = np.random.rand(num_nodes, 2)
    valids = world.is_valid_batch(X_cand)
    X = np.vstack([start, goal, X_cand[valids]])
    near_indices_list = []
    for i, x in enumerate(X):
        dists = np.linalg.norm(X - x, axis=1)
        neare_indices = np.where(dists < radius)[0]
        near_indices_list.append(neare_indices)
    return X, near_indices_list


def apply_astar(X, near_indices, ignore_indices: Optional[List[int]] = None, dijkstra: bool = False):
    open_set = {0}
    close_set = set()
    if ignore_indices is not None:
        close_set = set(ignore_indices)
        assert not 0 in close_set

    goal_pos = X[1]
    g_costs = np.inf * np.ones(X.shape[0])
    g_costs[0] = 0

    parents = np.ones(X.shape[0], dtype=int) * -1

    count_linalg = 0

    def compute_euclidean_distance(x1, x2):
        nonlocal count_linalg
        count_linalg += 1
        return np.linalg.norm(x1 - x2)

    if dijkstra:
        heuristic = lambda x: 0
    else:
        heuristic = lambda x: compute_euclidean_distance(x, goal_pos)

    count_iteration = 0
    while True:
        count_iteration += 1
        # find the best node in the open set wrt estimated cost
        best_node = None
        best_cost = np.inf
        for node_idx in open_set:
            f = g_costs[node_idx] + heuristic(X[node_idx])
            if f < best_cost:
                best_node = node_idx
                best_cost = f

        if best_node == 1:
            print(f"Found the goal in {count_iteration} iterations")
            # reverse trace the path
            idx_reverse_path = [1]
            while parents[idx_reverse_path[-1]] != 0:
                idx_reverse_path.append(parents[idx_reverse_path[-1]])
            idx_reverse_path.append(0)
            idx_reverse_path.reverse()
            print(f"number of linalg calls: {count_linalg}")
            return X[idx_reverse_path]

        # move the best node from the open set to the close set
        open_set.remove(best_node)
        close_set.add(best_node)

        # update the estimated cost of the neighbors of the best node
        for near_index in near_indices[best_node]:
            if near_index in close_set:
                continue
            if near_index not in open_set:
                open_set.add(near_index)
            g_cost_cand = g_costs[best_node] + compute_euclidean_distance(X[best_node], X[near_index])
            if g_cost_cand < g_costs[near_index]:
                g_costs[near_index] = g_cost_cand
                parents[near_index] = best_node


def find_ignorable_indices(X, path: np.ndarray, eps: float = 0.2):
    ignore_indices = []
    for i, x in enumerate(X):
        min_dist_to_path = np.min(np.linalg.norm(path - x, axis=1))
        if min_dist_to_path > eps:
            ignore_indices.append(i)
    return ignore_indices


if __name__ == "__main__":
    world = World()
    X, near_indices = build_prm_graph(world, np.array([0.1, 0.1]), np.array([0.5, 0.2]), 10000, radius=0.02)
    print("=> from scratch")
    path = apply_astar(X, near_indices, dijkstra=False)
    ignore_indices = find_ignorable_indices(X, path, eps=0.1)
    print("=> leveraging the previous path")
    path = apply_astar(X, near_indices, ignore_indices=ignore_indices, dijkstra=False)

    plot = False
    if plot:
        fig, ax = world.visualize()
        ax.scatter(X[:, 0], X[:, 1], s=5)
        for near_indices_i, x in zip(near_indices, X):
            for near_index in near_indices_i:
                ax.plot([x[0], X[near_index, 0]], [x[1], X[near_index, 1]], 'b-', linewidth=0.5)
        ax.plot(path[:, 0], path[:, 1], 'g-', linewidth=4)
        plt.show()
