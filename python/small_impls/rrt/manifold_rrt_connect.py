import argparse
import matplotlib.pyplot as plt
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Tuple

import numpy as np

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
    q_goal: Optional[np.ndarray]
    b_min: np.ndarray
    b_max: np.ndarray
    motion_step_box: np.ndarray
    nodes: List[Node]
    f_project: Callable[[np.ndarray], Optional[np.ndarray]]
    f_is_valid: Callable[[np.ndarray], bool]
    termination_hook: Optional[Callable[[], None]]
    config: ManifoldRRTConfig
    n_extension_trial: int

    def __init__(
        self,
        start: np.ndarray,
        goal: Optional[np.ndarray],
        b_min: np.ndarray,
        b_max: np.ndarray,
        motion_step_box: np.ndarray,
        f_project: Callable[[np.ndarray], Optional[np.ndarray]],
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
        self.motion_step_box = motion_step_box
        self.f_project = f_project
        self.f_is_valid = f_is_valid
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

    def extend(self, q_rand: np.ndarray, node_nearest: Optional[Node] = None) -> ExtensionResult:

        if self.termination_hook is not None:
            self.termination_hook()

        self.n_extension_trial += 1

        if node_nearest is None:
            node_nearest = self.find_nearest_node(q_rand)
        diff_ambient = q_rand - node_nearest.q

        if np.all(np.abs(diff_ambient) < self.motion_step_box):
            return ExtensionResult.REACHED

        shrink_motion_box = self.motion_step_box * self.config.motion_step_shring_rate
        diff_clamped = clamp(diff_ambient, shrink_motion_box)

        # check if projection successful
        q_new = self.f_project(node_nearest.q + diff_clamped)
        if q_new is None:
            return ExtensionResult.TRAPPED

        # check if q_new is inside configuration box space
        if np.any(q_new < self.b_min):
            return ExtensionResult.TRAPPED
        if np.any(q_new > self.b_max):

            return ExtensionResult.TRAPPED

        # check if motion step constraint is satisfied
        diff_actual = q_new - node_nearest.q

        if np.linalg.norm(diff_actual) < 1e-6:
            return ExtensionResult.TRAPPED

        if not np.all(np.abs(diff_actual) < self.motion_step_box):
            return ExtensionResult.TRAPPED

        # check if inequality constraint is satisfied
        if not self.f_is_valid(q_new):
            return ExtensionResult.TRAPPED

        new_node = Node(q_new, node_nearest)
        self.nodes.append(new_node)
        return ExtensionResult.ADVANCED

    def connect(self, q_target: np.ndarray) -> Optional[Node]:
        # reutrn connceted Node if connected. Otherwise return None

        self.find_nearest_node(q_target)
        result = self.extend(q_target, None)
        if result != ExtensionResult.ADVANCED:
            return None

        # because last result is advanced
        while True:
            result = self.extend(q_target, self.nodes[-1])
            if result == ExtensionResult.TRAPPED:
                return None
            if result == ExtensionResult.REACHED:
                return self.nodes[-1]

    def visualize(self, fax):
        if self.dof == 2:
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
        if self.dof == 3:
            if fax is None:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")
            else:
                fig, ax = fax
            Q = np.array([n.q for n in self.nodes])

            ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], s=2)
            for n in self.nodes:
                if n.node_parent is None:
                    continue
                q = n.q
                q_parent = n.node_parent.q
                ax.plot(
                    [q_parent[0], q[0]], [q_parent[1], q[1]], [q_parent[2], q[2]], c="red", linewidth=1
                )


class ManifoldRRTConnect:
    """
    Class for planning a path between two configurations using two ManifoldRRT trees.
    (generated by chat-gpt)

    Parameters
    ----------
    q_start : np.ndarray
        The starting configuration.
    q_goal : np.ndarray
        The goal configuration.
    b_min : np.ndarray
        The minimum values of the configuration space bounds.
    b_max : np.ndarray
        The maximum values of the configuration space bounds.
    motion_step_box : np.ndarray
        The size of the motion step box for the RRT algorithm.
    f_project : Callable[[np.ndarray], Optional[np.ndarray]]
        A function that projects a given configuration onto the manifold.
        return None if projection failed
    f_is_valid : Callable[[np.ndarray], bool]
        A function that checks if a given configuration is valid.
    config : Config, optional
        The configuration object for the RRT algorithm, by default Config().
    """

    rrt_start: ManifoldRRT
    rrt_goal: ManifoldRRT
    connection: Optional[Tuple[Node, Node]] = None
    n_extension_trial: int

    def __init__(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        b_min: np.ndarray,
        b_max: np.ndarray,
        motion_step_box: np.ndarray,
        f_project: Callable[[np.ndarray], Optional[np.ndarray]],
        f_is_valid: Callable[[np.ndarray], bool],
        config: ManifoldRRTConfig = ManifoldRRTConfig(),
    ):
        self.rrt_start = ManifoldRRT(
            q_start, None, b_min, b_max, motion_step_box, f_project, f_is_valid, None, config
        )
        self.rrt_goal = ManifoldRRT(
            q_goal, None, b_min, b_max, motion_step_box, f_project, f_is_valid, None, config
        )

        def termination_hook():
            n_total = self.rrt_start.n_extension_trial + self.rrt_goal.n_extension_trial
            if n_total > config.n_max_call:
                raise TerminationException

        self.rrt_start.termination_hook = termination_hook
        self.rrt_goal.termination_hook = termination_hook

        self.connection = None

    def solve(self) -> bool:
        try:
            extend_start_tree = True
            while True:
                if extend_start_tree:
                    rrt_a = self.rrt_start
                    rrt_b = self.rrt_goal
                    extend_start_tree = False
                else:
                    rrt_a = self.rrt_goal
                    rrt_b = self.rrt_start
                    extend_start_tree = True
                q_rand = rrt_a.sample()
                res = rrt_a.extend(q_rand)
                if res == ExtensionResult.ADVANCED:
                    node_target = rrt_a.nodes[-1]
                    node = rrt_b.connect(node_target.q)
                    if node is not None:
                        if extend_start_tree:
                            self.connection = (node, node_target)
                        else:
                            self.connection = (node_target, node)
                        return True
        except TerminationException:
            return False
        return False

    def get_solution(self) -> np.ndarray:
        assert self.connection is not None
        node = self.connection[0]
        q_seq_start = [node.q]
        while True:
            node = node.node_parent
            if node is None:
                break
            q_seq_start.append(node.q)

        node = self.connection[1]
        q_seq_goal = [node.q]
        while True:
            node = node.node_parent
            if node is None:
                break
            q_seq_goal.append(node.q)

        q_seq = list(reversed(q_seq_start)) + q_seq_goal
        return np.array(q_seq)


if __name__ == "__main__":
    # argparse to select mode
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="manifold")
    args = parser.parse_args()

    if args.mode == "manifold":
        def project(q: np.ndarray) -> Optional[np.ndarray]:
            return q / np.linalg.norm(q)

        def is_valid(q: np.ndarray) -> bool:
            if abs(q[0]) > 0.2:
                return True
            if abs(q[1]) < 0.2:
                return True
            return False

        start = np.array([-1, 0, 0])
        goal = np.array([+1, 0, 0])
        b_min = -np.ones(3) * 1.5
        b_max = +np.ones(3) * 1.5
        motion_step_box = np.ones(3) * 0.2
        conf = ManifoldRRTConfig(1000)
        bitree = ManifoldRRTConnect(
            start, goal, b_min, b_max, motion_step_box, project, is_valid, config=conf
        )
        import time

        ts = time.time()
        res = bitree.solve()
        print(time.time() - ts)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        bitree.rrt_start.visualize((fig, ax))
        bitree.rrt_goal.visualize((fig, ax))
        Q = bitree.get_solution()
        ax.plot(Q[:, 0], Q[:, 1], Q[:, 2], c="k", linewidth=3)
        plt.show()
    elif args.mode == "normal":

        def project(q: np.ndarray) -> Optional[np.ndarray]:
            return q

        def is_valid(q: np.ndarray) -> bool:
            # check if outside of circle with radius 0.3 and center (0.5, 0.5)
            return np.linalg.norm(q - np.array([0.5, 0.5])) > 0.3

        start = np.ones(2) * 0.1
        goal = np.ones(2) * 0.9
        b_min = np.zeros(2)
        b_max = np.ones(2)
        motion_step_box = np.ones(2) * 0.1
        conf = ManifoldRRTConfig(1000)
        rrt = ManifoldRRT(start, goal, b_min, b_max, motion_step_box, project, is_valid, config=conf)

        def is_close_to_goal(q: np.ndarray) -> bool:
            # return np.linalg.norm(q - goal) < 0.02
            # check using motion_step_box
            diff = q - goal
            return np.all(np.abs(diff) < motion_step_box)


        for _ in range(1000):
            ext_res = rrt.step()
            print(ext_res)
            # check if the latest node reaches goal
            if is_close_to_goal(rrt.nodes[-1].q):
                print("solved")
                break
        rrt.visualize(None)
        plt.show()
    else:
        assert False
