import copy
import numpy as np
import torch as th
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Union, Optional
from gymnasium import Env
from gymnasium.spaces import Box

from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import ActorCriticPolicy


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
        # return abs(np.cos(theta) - (-1)) < 0.04 and abs(theta_dot) < 0.1
        return abs(np.cos(theta) - (-1)) < 0.02 and abs(theta_dot) < 0.03

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

    def __call__(self, state: np.ndarray) -> float:
        x, x_dot, theta, theta_dot = state
        m, M, l, g = self.model_param.m, self.model_param.M, self.model_param.l, self.model_param.g
        s, c = np.sin(theta), np.cos(theta)
        E_swing = 0.5 * m * l ** 2 * theta_dot ** 2 + m * g * l * (1 - c)
        E_swing_d = 2 * m * g * l
        u = self.alpha * theta_dot * c * (E_swing - E_swing_d)
        f = (M + m * s ** 2) * u - (m * l * s * theta_dot ** 2 + m * g * s * c)
        return f    


class CartpoleEnv(Env):
    system: Cartpole
    observation_space: Box
    action_space: Box
    # observation_space: spaces.Space[ObsType]

    def __init__(self, model_param: ModelParameter = ModelParameter()):
        self.system = Cartpole(np.array([0.0, 0.0, 0.0, 0.0]), model_param)
        obs_high = np.array([np.inf, np.inf, np.inf, np.inf]) 
        obs_low = -1 * obs_high
        self.observation_space = Box(obs_low, obs_high, dtype=np.float32)

        act_high = np.array([40.0])
        act_low = -1 * act_high
        self.action_space = Box(act_low, act_high, dtype=np.float32)
        self.reset()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:  # type: ignore
        self.system.state = np.random.randn(4) * np.array([0.1, 0.1, 0.1, 0.1])
        return self.system.state, {}

    def step(
        self, action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.system.step(action[0])
        obs = self.system.state
        is_termianted = self.system.is_uplight()
        if is_termianted:
            print("uplight")
        reward = 1.0 if is_termianted else 0.0
        return obs, reward, is_termianted, False, {}


class PartiallyAnalyticPolicy(ActorCriticPolicy):
    controller: EnergyShapingController

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.controller = EnergyShapingController(ModelParameter())

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        action_residual, values, log_prob = super().forward(obs, deterministic)
        obs_np = obs.cpu().detach().numpy()[0]
        f: float = self.controller(obs_np)
        action = action_residual + th.from_numpy(np.array([f])).to(self.device)
        return action, values, log_prob


if __name__ == "__main__":
    env = CartpoleEnv()
    model = PPO(PartiallyAnalyticPolicy, env, verbose=1)
    model.learn(total_timesteps=20000)
