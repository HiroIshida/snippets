import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional

import numpy as np

from lib import TrajectoryPiece, optimal_cost


@dataclass
class Controller:
    u_min: np.ndarray
    u_max: np.ndarray


@dataclass
class State:
    s: np.ndarray

    @property
    def x(self) -> np.ndarray:
        return self.s[:2]

    @property
    def v(self) -> np.ndarray:
        return self.s[2:]

    def propagate(self, u: np.ndarray, dt: float) -> "State":
        x = self.x + self.v * dt + u * dt**2 / 2
        v = self.v + u * dt
        return State(np.concatenate([x, v]))

    def is_forward_reachable(self, state: "State", ctrl: Controller, dt: float) -> bool:
        x_max = self.x + self.v * dt + ctrl.u_max * dt**2 / 2
        x_min = self.x + self.v * dt - ctrl.u_min * dt**2 / 2
        if np.any(state.x > x_max) or np.any(state.x < x_min):
            return False
        v_max = self.v + ctrl.u_max * dt
        v_min = self.v - ctrl.u_min * dt
        if np.any(state.v > v_max) or np.any(state.v < v_min):
            return False
        return True


@dataclass
class StateBound:
    x_min: np.ndarray
    x_max: np.ndarray
    v_min: np.ndarray
    v_max: np.ndarray

    def sample(self) -> State:
        x = np.random.uniform(self.x_min, self.x_max)
        v = np.random.uniform(self.v_min, self.v_max)
        return State(np.concatenate([x, v]))

    def is_inside(self, state: State) -> bool:
        if np.any(state.x > self.x_max) or np.any(state.x < self.x_min):
            return False
        if np.any(state.v > self.v_max) or np.any(state.v < self.v_min):
            return False
        return True


@dataclass
class Node:
    state: State
    duration_to_parent: Optional[float] = None
    parent: Optional["Node"] = None


class Status(Enum):
    ADVANED = 0
    TRAPPED = 1
    REACHED = 2


class RRT:
    start: State
    goal: State
    is_obstacle_free: Callable[[State], bool]
    state_bound: StateBound
    ctrl: Controller
    dt_extend: float
    dt_resolution: float
    nodes: List[Node]

    def __init__(
        self,
        start: State,
        goal: State,
        is_obstacle_free: Callable[[State], bool],
        state_bound: StateBound,
        ctrl: Controller,
        dt_extend: float,
        dt_resolution: float = 0.1,
    ):
        self.start = start
        self.goal = goal
        self.is_obstacle_free = is_obstacle_free
        self.state_bound = state_bound
        self.ctrl = ctrl
        self.dt_extend = dt_extend
        self.dt_resolution = dt_resolution
        self.nodes = [Node(start)]

    def is_valid(self, state: State) -> bool:
        return self.state_bound.is_inside(state) and self.is_obstacle_free(state)

    def nearest_node(self, state_query: State) -> Node:
        min_dist = np.inf
        nearest_node = None
        for node in self.nodes:
            dist = float(np.linalg.norm(node.state.s - state_query.s))
            if dist < min_dist:
                min_dist = dist
                nearest_node = node
        assert nearest_node is not None
        return nearest_node

    def extend(self, state_query: State) -> Status:
        nearest_node = self.nearest_node(state_query)
        traj_piece = TrajectoryPiece(nearest_node.state.s, state_query.s)  # optimal trajectory
        status = Status.TRAPPED
        state_to_added = None
        # time list from dt_resolution to dt_extend in one-liner
        time_list = list(np.arange(self.dt_resolution, self.dt_extend, self.dt_resolution))
        time_list.append(self.dt_extend)
        for t in time_list:
            state = traj_piece.interpolate(t)
            if not self.is_valid(State(state)):
                if state_to_added is not None:
                    status = Status.ADVANED
                    new_node = Node(State(state_to_added), t, nearest_node)
                    self.nodes.append(new_node)
                else:
                    status = Status.TRAPPED
                return status
            state_to_added = state

        assert state_to_added is not None
        status = Status.REACHED
        new_node = Node(State(state_to_added), self.dt_extend, nearest_node)
        self.nodes.append(new_node)
        return status

    def solve(self, n_max_iter: int) -> bool:
        for i in range(n_max_iter):
            print(i)
            state_query = self.state_bound.sample()
            status = self.extend(state_query)
            if status != Status.TRAPPED:
                # try to connect the latest node to the goal
                time_optimal, _ = optimal_cost(self.nodes[-1].state.s, self.goal.s)
                if time_optimal < self.dt_extend:
                    traj_piece = TrajectoryPiece(
                        self.nodes[-1].state.s, self.goal.s
                    )  # optimal connection

                    def connectable():
                        time_list = list(
                            np.arange(self.dt_resolution, time_optimal, self.dt_resolution)
                        )
                        for t in time_list:
                            state = traj_piece.interpolate(t)
                            if not self.is_valid(State(state)):
                                return False
                        return True

                    if connectable():
                        new_node = Node(self.goal, time_optimal, self.nodes[-1])
                        self.nodes.append(new_node)
                        return True
        return False

    def get_solution(self) -> List[Node]:
        node: Optional[Node] = self.nodes[-1]
        solution = []
        while node is not None:
            solution.append(node)
            node = node.parent
        solution.reverse()
        return solution


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    def is_valid(state: State) -> bool:
        # circle obstacle at (0.5, 0.5) with radius 0.1
        if np.linalg.norm(state.x - np.array([0.5, 0.5])) < 0.4:
            return False
        return True

    start = State(np.array([0.1, 0.1, 0, 0]))
    goal = State(np.array([0.9, 0.9, 0, 0]))
    u_abs = 0.3
    state_bound = StateBound(
        np.array([0, 0]), np.array([1, 1]), np.array([-u_abs, -u_abs]), np.array([u_abs, u_abs])
    )
    ctrl = Controller(-u_abs * np.ones(2), u_abs * np.ones(2))
    rrt = RRT(start, goal, is_valid, state_bound, ctrl, 1.0, 0.1)
    ts = time.time()
    is_solved = rrt.solve(2000)
    print(time.time() - ts)
    # # assert is_solved
    # solution = rrt.get_solution()

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect="equal")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, edgecolor="black"))

    # plot all node by scatter
    S = np.array([node.state.s for node in rrt.nodes])
    ax.scatter(S[:, 0], S[:, 1], c="blue", s=10)

    # plot all connection
    for node in rrt.nodes:
        if node.parent is not None:
            traj_piece = TrajectoryPiece(node.parent.state.s, node.state.s, node.duration_to_parent)
            time_list = list(np.arange(0, node.duration_to_parent, rrt.dt_resolution))
            time_list.append(node.duration_to_parent)
            states = list(map(traj_piece.interpolate, time_list))
            # plot all edges
            for i in range(len(states) - 1):
                ax.plot(
                    [states[i][0], states[i + 1][0]],
                    [states[i][1], states[i + 1][1]],
                    c="black",
                    lw=0.5,
                )

    solution = rrt.get_solution()
    # plot solution by scatter
    x = [state.state.x[0] for state in solution]
    y = [state.state.x[1] for state in solution]
    ax.scatter(x, y, c="red", s=50)

    # plot edges of solution using interpolations
    for i in range(len(solution) - 1):
        traj_piece = TrajectoryPiece(
            solution[i].state.s, solution[i + 1].state.s, solution[i + 1].duration_to_parent
        )
        time_list = list(np.arange(0, solution[i + 1].duration_to_parent, rrt.dt_resolution))
        time_list.append(solution[i + 1].duration_to_parent)
        states = list(map(traj_piece.interpolate, time_list))
        for i in range(len(states) - 1):
            ax.plot(
                [states[i][0], states[i + 1][0]], [states[i][1], states[i + 1][1]], c="red", lw=2
            )

    # plot square patch of the world [0, 1] x [0, 1]
    ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, edgecolor="black"))

    # plot the circle indicating the obstacle adding to ax
    circle = plt.Circle((0.5, 0.5), 0.4, color="black", fill=False)
    ax.add_artist(circle)

    plt.show()
