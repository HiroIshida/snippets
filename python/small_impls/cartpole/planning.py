import numpy as np
import tqdm
import argparse
import matplotlib.pyplot as plt
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Tuple

import numpy as np

from cartpole import Cartpole, ModelParameter, EnergyShapingController

def clamp(v, width):
    if np.all(v == 0):
        return np.zeros_like(v)
    with np.errstate(divide='ignore', invalid='ignore'):  # Handle division by zero
        scale_factors = np.abs(width / v)
    min_scale = np.nanmin(scale_factors)
    return v * min_scale


class ExtensionResult(Enum):
    REACHED = 0
    ADVANCED = 1
    TRAPPED = 2


@dataclass
class ManifoldRRTConfig:
    n_max_call: int = 2000
    motion_step_shring_rate: float = 0.5


@dataclass
class Node:
    q: np.ndarray
    node_parent: Optional["Node"] = None


class TerminationException(Exception):
    ...


class ManifoldRRT(ABC):
    q_goal: np.ndarray
    b_min: np.ndarray
    b_max: np.ndarray
    nodes: List[Node]
    f_prop: Callable[[np.ndarray, float], np.ndarray]
    f_heuristic: Callable[[np.ndarray], float]
    f_is_valid: Callable[[np.ndarray], bool]
    termination_hook: Optional[Callable[[], None]]
    config: ManifoldRRTConfig
    n_extension_trial: int

    def __init__(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        b_min: np.ndarray,
        b_max: np.ndarray,
        f_prop: Callable[[np.ndarray], np.ndarray],
        f_heuristic: Callable[[np.ndarray], np.ndarray],
        f_is_valid: Callable[[np.ndarray], bool],
        termination_hook: Optional[Callable[[], None]] = None,
        config: ManifoldRRTConfig = ManifoldRRTConfig(),
    ):

        assert f_is_valid(start)
        if goal is not None:
            f_is_valid(goal)
        self.q_goal = goal
        self.b_min = b_min
        self.b_max = b_max
        self.nodes = [Node(start, None)]
        self.f_prop = f_prop
        self.f_is_valid = f_is_valid
        self.f_heuristic = f_heuristic
        self.termination_hook = termination_hook
        self.config = config
        self.n_extension_trial = 0

    @property
    def start_node(self) -> Optional[Node]:
        return self.nodes[0]

    @property
    def dof(self) -> int:
        return len(self.b_min)

    def sample(self) -> np.ndarray:
        q = np.random.rand(self.dof) * (self.b_max - self.b_min) + self.b_min
        return q

    def find_nearest_node(self, q: np.ndarray) -> Node:
        min_idx = np.argmin([np.linalg.norm(q - n.q) for n in self.nodes])
        return self.nodes[min_idx]

    def step(self) -> ExtensionResult:
        q_rand = self.sample()
        return self.extend(q_rand)

    def extend(self, q_rand: np.ndarray) -> ExtensionResult:

        if self.termination_hook is not None:
            self.termination_hook()

        self.n_extension_trial += 1

        node_nearest = self.find_nearest_node(q_rand)

        q_cand_list = []
        for _ in range(20):
            u = self.f_heuristic(node_nearest.q)
            width = 0.5
            u += np.random.rand() * width - 0.5 * width
            q_cand = self.f_prop(node_nearest.q, u)
            if np.all(q_cand > self.b_min) and np.all(q_cand < self.b_max):
                q_cand_list.append(q_cand)
        if len(q_cand_list) == 0:
            return ExtensionResult.TRAPPED

        q_cand = min(q_cand_list, key=lambda q: np.linalg.norm(q - q_rand))
        new_node = Node(q_cand, node_nearest)
        self.nodes.append(new_node)
        return ExtensionResult.ADVANCED

    def visualize(self, fax):
        if fax is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = fax
        Q = np.array([n.q for n in self.nodes])

        ax.scatter(Q[:, 0], Q[:, 1], s=2)
        for n in self.nodes:
            if n.node_parent is None:
                continue
            q = n.q
            q_parent = n.node_parent.q
            ax.plot([q_parent[0], q[0]], [q_parent[1], q[1]], c="red", linewidth=1)


if __name__ == "__main__":
    model_param = ModelParameter()

    def f_prop(state: np.ndarray, f: float, dt: float = 0.05):
        x, x_dot, theta, theta_dot = state
        m, M, l, g = model_param.m, model_param.M, model_param.l, model_param.g
        x_acc = (f + m * np.sin(theta) * (l * theta_dot ** 2 + g * np.cos(theta))) / (M + m * np.sin(theta) ** 2)
        theta_acc = -1 * (np.cos(theta) * x_acc + g * np.sin(theta)) / l
        x_dot += x_acc * dt
        theta_dot += theta_acc * dt
        x += x_dot * dt
        theta += theta_dot * dt
        return np.array([x, x_dot, theta, theta_dot])

    cont = EnergyShapingController(model_param, 0.1)

    b_max = np.array([10.0, 3.0, np.pi * 1.5, np.pi * 2.5])
    b_min = -1 * b_max
    conf = ManifoldRRTConfig(n_max_call=1000)
    rrt = ManifoldRRT(np.zeros(4), None, b_min, b_max, f_prop, cont, lambda q: True, config=conf)
    for _ in tqdm.tqdm(range(5000)):
        ret = rrt.step()
        q_latest = rrt.nodes[-1].q
        theta = q_latest[2]
        theta_dot = q_latest[3]
        if np.abs(np.cos(theta) - (-1)) < 0.02 and np.abs(theta_dot) < 0.3:
            print("found")
            break
    rrt.visualize(None)
    # plot the latest node as a huge marker
    plt.scatter(q_latest[0], q_latest[1], s=100, c="red")
    plt.show()
