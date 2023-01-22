# this file is fetched from https://github.com/zuoxingdong/mazelab
import time
from dataclasses import dataclass
import numpy as np
from typing import Tuple


def random_maze(width=81, height=51, complexity=.75, density=.75):
    r"""Generate a random maze array. 
    
    It only contains two kind of objects, obstacle and free space. The numerical value for obstacle
    is ``1`` and for free space is ``0``. 
    
    Code from https://en.wikipedia.org/wiki/Maze_generation_algorithm
    """
    # Only odd shapes
    shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)
    # Adjust complexity and density relative to maze size
    complexity = int(complexity * (5 * (shape[0] + shape[1])))
    density    = int(density * ((shape[0] // 2) * (shape[1] // 2)))
    # Build actual maze
    Z = np.zeros(shape, dtype=bool)
    # Fill borders
    Z[0, :] = Z[-1, :] = 1
    Z[:, 0] = Z[:, -1] = 1
    # Make aisles
    for i in range(density):
        x, y = np.random.randint(0, shape[1]//2 + 1) * 2, np.random.randint(0, shape[0]//2 + 1) * 2
        Z[y, x] = 1
        for j in range(complexity):
            neighbours = []
            if x > 1:             neighbours.append((y, x - 2))
            if x < shape[1] - 2:  neighbours.append((y, x + 2))
            if y > 1:             neighbours.append((y - 2, x))
            if y < shape[0] - 2:  neighbours.append((y + 2, x))
            if len(neighbours):
                y_,x_ = neighbours[np.random.randint(0, len(neighbours))]
                if Z[y_, x_] == 0:
                    Z[y_, x_] = 1
                    Z[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                    x, y = x_, y_
                    
    return Z.astype(int)


@dataclass
class MazeSDF:
    maze: np.ndarray

    @staticmethod
    def compute_box_sdf(points: np.ndarray, origin: np.ndarray, width: np.ndarray):
        half_extent = width * 0.5
        pts_from_center = points - origin[None, :]
        sd_vals_each_axis = np.abs(pts_from_center) - half_extent[None, :]

        positive_dists_each_axis = np.maximum(sd_vals_each_axis, 0.0)
        positive_dists = np.sqrt(np.sum(positive_dists_each_axis**2, axis=1))

        negative_dists_each_axis = np.max(sd_vals_each_axis, axis=1)
        negative_dists = np.minimum(negative_dists_each_axis, 0.0)

        sd_vals = positive_dists + negative_dists
        return sd_vals

    def __call__(self) -> np.ndarray:
        box_width = np.ones(2) / self.maze.shape
        index_pair_list = [np.array(e) for e in zip(*np.where(self.maze == 1))]
        mat = np.ones((100, 100)) * np.inf

        N = 100
        b_min = np.zeros(2)
        b_max = np.ones(2)
        xlin, ylin = [np.linspace(b_min[i], b_max[i], N) for i in range(2)]
        X, Y = np.meshgrid(xlin, ylin)
        pts = np.array(list(zip(X.flatten(), Y.flatten())))

        box_width = np.ones(2) / self.maze.shape

        vals = np.ones(len(pts)) * np.inf

        for index_pair in index_pair_list:
            pos = box_width * np.array(index_pair) + box_width * 0.5
            vals_cand = self.compute_box_sdf(pts, pos, box_width)
            vals = np.minimum(vals, vals_cand)

        return vals.reshape(N, N)


if __name__ == "__main__":
    N = 12
    Z = random_maze(N, N, density=0.4, complexity=0.1)
    sdf = MazeSDF(Z)
    ts = time.time()
    mat = sdf()
    print(time.time() - ts)

    import matplotlib.pyplot as plt
    plt.imshow(mat)
    plt.show()
