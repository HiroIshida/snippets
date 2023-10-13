import copy
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List

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
        return abs(np.cos(theta) - (-1)) < 0.04 and abs(theta_dot) < 0.1

    def render(self, ax):
        print(self.state)
        x, x_dot, theta, theta_dot = self.state
        m, M, l, g = self.model_param.m, self.model_param.M, self.model_param.l, self.model_param.g
        cart = plt.Circle((x, 0), 0.2, color="black")
        ax.add_patch(cart)
        pole = plt.Line2D((x, x + l * np.sin(theta)), (0, - l * np.cos(theta)), color="black")
        ax.add_line(pole)


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
    f_history = []

    for _ in range(2000):
        f = controller(system.state)
        f_history.append(f)
        system.step(f)
        if system.is_uplight():
            print("uplight")
            break

    # fig, ax = plt.subplots()
    # # set xlim
    # ax.set_xlim(-40, 40)
    # ax.set_ylim(-2, 2)
    # system.render(ax)
    # # equal axis
    # ax.set_aspect('equal')
    # plt.show()
    

    history = np.array(system.history)
    phase_state_of_pole = history[:, 2:4]
    # plt.plot(phase_state_of_pole[:, 0], phase_state_of_pole[:, 1])
    plt.plot(history[:, 2])
    plt.show()
