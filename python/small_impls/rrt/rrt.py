import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import Optional, Protocol, Generic, TypeVar, List, Set


class MazeLike(Protocol):

    def sample(self) -> np.ndarray:
        ...

    def is_colliding(self, state: np.ndarray) -> bool:
        ...

@dataclass
class RRTConfig:
    eps: float = 0.05
    n_max_iter: int = 30000
    goal_margin: float = 0.05

class RRT:
    maze: MazeLike
    goal: np.ndarray
    sample_count: int
    config: RRTConfig
    _samples: np.ndarray
    _parent_indices: np.ndarray
    is_terminated: bool

    def __init__(self, start: np.ndarray, goal: np.ndarray, maze: MazeLike, config: Optional[RRTConfig] = None):
        if config is None:
            config = RRTConfig()
        self.config = config

        # reserve memory to avoid dynamic allocation in each iteration
        n_dof = len(start)
        self.maze = maze
        self._samples = np.zeros((config.n_max_iter, n_dof))
        self._parent_indices = np.zeros(config.n_max_iter, dtype=int)
        self._samples[0] = start
        self.sample_count = 1
        self.goal = goal
        self.is_terminated = False

    @property
    def samples(self) -> np.ndarray:
        return self._samples[:self.sample_count]

    @property
    def parent_indices(self) -> np.ndarray:
        return self._parent_indices[:self.sample_count]

    @staticmethod
    def normalize(vec: np.ndarray) -> np.ndarray:
        return vec/np.linalg.norm(vec)

    def extend(self) -> bool:
        assert not self.is_terminated
        state_rand = self.maze.sample()
        dists = np.sqrt(np.sum((self.samples - state_rand) ** 2, axis=1))
        idx_nearest = np.argmin(dists)
        state_nearest = self.samples[idx_nearest]

        # extend
        unit_vec = self.normalize(state_rand - state_nearest)
        state_new = state_nearest + min(dists[idx_nearest], self.config.eps) * unit_vec

        ## update tree
        if not self.maze.is_colliding(state_new):
            idx_new = self.sample_count
            self._samples[idx_new] = state_new
            self._parent_indices[idx_new] = idx_nearest
            self.sample_count += 1

            if np.linalg.norm(self.goal - state_new) < self.config.goal_margin:
                self.is_terminated = True
                return True
        return False

    def get_solution(self) -> List[np.ndarray]:
        idx = self.sample_count - 1
        trajectory = [self.goal, self.samples[idx]]
        while idx != 0:
            idx = self.parent_indices[idx]
            trajectory.append(self.samples[idx])
        return trajectory

    def visualize(self):
        fig, ax = plt.subplots()

        # visualize nodes
        ax.scatter(self.samples[:, 0], self.samples[:, 1], c="black", s=5)
        
        # visualize edge
        for idx, state in enumerate(self.samples):
            idx_parent = self.parent_indices[idx]
            parent = self.samples[idx_parent]
            arr = np.stack([state, parent])
            ax.plot(arr[:, 0], arr[:, 1], color="red", linewidth=0.5)

        # visualize solution
        if self.is_terminated:
            trajectory = self.get_solution()
            arr = np.array(trajectory)
            ax.plot(arr[:, 0], arr[:, 1], color="blue", linewidth=1.0)


if __name__ == "__main__":
    class SimpleWorld:

        def sample(self) -> np.ndarray:
            return np.random.rand(2)

        def is_colliding(self, state: np.ndarray) -> bool:
            dist = float(np.linalg.norm(state - np.array([0.5, 0.5])))
            return dist < 0.2

    rrt = RRT(np.array([0.1, 0.1]), np.array([0.9, 0.9]), SimpleWorld())
    while not rrt.extend():
        pass
    rrt.visualize()
    plt.show()
