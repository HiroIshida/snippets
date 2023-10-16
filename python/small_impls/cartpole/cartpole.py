import copy
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List
from control import lqr

@dataclass
class ModelParameter:
    m: float = 1.0
    M: float = 1.0
    l: float = 1.0
    g: float = 9.8


class Cartpole:
    state: np.ndarray
    model_param: ModelParameter
    history: List[np.ndarray]

    def __init__(self, state: np.ndarray, model_param: ModelParameter = ModelParameter()):
        self.state = state
        self.model_param = model_param
        self.history = []

    def step(self, f: float, dt: float = 0.05):
        x, x_dot, theta, theta_dot = self.state
        m, M, l, g = self.model_param.m, self.model_param.M, self.model_param.l, self.model_param.g
        x_acc = (f + m * np.sin(theta) * (l * theta_dot ** 2 + g * np.cos(theta))) / (M + m * np.sin(theta) ** 2)
        theta_acc = -1 * (np.cos(theta) * x_acc + g * np.sin(theta)) / l
        x_dot += x_acc * dt
        theta_dot += theta_acc * dt
        x += x_dot * dt
        theta += theta_dot * dt
        self.state = np.array([x, x_dot, theta, theta_dot])
        self.history.append(self.state)

    def is_uplight(self) -> bool:
        x, x_dot, theta, theta_dot = self.state
        return abs(np.cos(theta) - (-1)) < 0.3 and abs(theta_dot) < 0.5

    def render(self, ax):
        print(self.state)
        x, x_dot, theta, theta_dot = self.state
        m, M, l, g = self.model_param.m, self.model_param.M, self.model_param.l, self.model_param.g
        cart = plt.Circle((x, 0), 0.2, color="black")
        ax.add_patch(cart)
        pole = plt.Line2D((x, x + l * np.sin(theta)), (0, - l * np.cos(theta)), color="black")
        ax.add_line(pole)


@dataclass
class LQRController:
    model_param: ModelParameter
    A: np.ndarray
    B: np.ndarray

    def __init__(self, model_param: ModelParameter):
        m, M, l, g = model_param.m, model_param.M, model_param.l, model_param.g

        def theta_acc(theta, theta_dot, u) -> float:
            x_acc = (u + m * np.sin(theta) * (l * theta_dot ** 2 + g * np.cos(theta))) / (M + m * np.sin(theta) ** 2)
            theta_acc = -1 * (np.cos(theta) * x_acc + g * np.sin(theta)) / l
            return theta_acc

        eps = 1e-6
        # linealize the system at the upright position
        # at the upright position, theta = pi, theta_dot = 0 and we done care about x and x_dot but 
        # so differentiate the theta_acc dynamics
        A = np.zeros((2, 2))
        A[0, 0] = 0
        A[0, 1] = 1.0
        A[1, 0] = (theta_acc(np.pi + eps, 0.0, 0.0) - theta_acc(np.pi, 0.0, 0.0)) / eps
        A[1, 1] = (theta_acc(np.pi, eps, 0.0) - theta_acc(np.pi, 0.0, 0.0)) / eps
        B = np.zeros((2, 1))
        B[1, 0] = (theta_acc(np.pi, 0.0, 0.0 + eps) - theta_acc(np.pi, 0.0, 0.0)) / eps
        self.A = A
        self.B = B

    def __call__(self, state: np.ndarray):
        x, x_dot, theta, theta_dot = state
        Q = np.eye(2)
        R = np.eye(1) * 0.1
        K, _, _ = lqr(self.A, self.B, Q, R)
        # target theta is pi * 2 * n * pi
        target_cand = [-3 * np.pi, -np.pi, np.pi, 3 * np.pi]
        target = min(target_cand, key=lambda x: abs(x - theta))
        u = -K @ np.array([theta - target, theta_dot])
        return u


# Chung, Chung Choo, and John Hauser. "Nonlinear control of a swinging pendulum." automatica 31.6 (1995): 851-862.
class EnergyShapingController: 
    model_param: ModelParameter
    alpha: float

    def __init__(self, model_param: ModelParameter, alpha: float = 0.1):
        self.model_param = model_param
        self.alpha = alpha

    def __call__(self, state: np.ndarray):
        x, x_dot, theta, theta_dot = state
        m, M, l, g = self.model_param.m, self.model_param.M, self.model_param.l, self.model_param.g
        s, c = np.sin(theta), np.cos(theta)
        E_swing = 0.5 * m * l ** 2 * theta_dot ** 2 + m * g * l * (1 - c)
        E_swing_d = 2 * m * g * l
        u = self.alpha * theta_dot * c * (E_swing - E_swing_d)
        f = (M + m * s ** 2) * u - (m * l * s * theta_dot ** 2 + m * g * s * c)
        return f    


if __name__ == "__main__":
    model_actual = ModelParameter()
    model_est = copy.deepcopy(model_actual)
    model_est.m = 0.98
    system = Cartpole(np.array([0.0, 0.0, 0.3, 0.0]))
    controller = EnergyShapingController(model_est)
    lqr_controller = LQRController(model_est)
    f_history = []

    for _ in range(2000):
        f = controller(system.state)
        f_history.append(f)
        system.step(f)
        if system.is_uplight():
            print("uplight")
            break

    for _ in range(2000):
        f = lqr_controller(system.state)[0]
        print(f)
        f_history.append(f)
        system.step(f)

    history = np.array(system.history)
    plt.plot(history[:, 2], history[:, 3])
    plt.plot(history[-1, 2], history[-1, 3], "o", markersize=10)
    print(history[-1, :])
    plt.show()
