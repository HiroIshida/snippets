import argparse
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
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from sb3_contrib import TRPO

@dataclass
class ModelParameter:
    m: float = 1.0
    M: float = 1.0
    l: float = 1.0
    g: float = 9.8

    @classmethod
    def create_random(cls) -> "ModelParameter":
        return cls(
            m=1.0 + np.random.randn() * 0.1,
            M=1.0 + np.random.randn() * 0.1,
            l=1.0 + np.random.randn() * 0.1,
            g=9.8,
        )


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
        # return abs(np.cos(theta) - (-1)) < 0.01 and abs(theta_dot) < 0.01

    def render(self, ax):
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
    randomize_model_param: bool
    num_uplight: int
    episode_count: int

    def __init__(self, model_param: ModelParameter = ModelParameter(), randomize_model_param: bool = True):
        self.system = Cartpole(np.array([0.0, 0.0, 0.0, 0.0]), model_param)
        obs_high = np.array([np.inf, np.inf, np.inf, np.inf]) 
        obs_low = -1 * obs_high
        self.observation_space = Box(obs_low, obs_high, dtype=np.float32)

        act_high = np.array([50.0])
        act_low = -1 * act_high
        self.action_space = Box(act_low, act_high, dtype=np.float32)
        self.num_uplight = 0
        self.randomize_model_param = randomize_model_param
        self.episode_count = 0
        self.reset()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:  # type: ignore
        self.episode_count = 0
        self.system.state = np.random.randn(4) * np.array([0.1, 0.1, 0.1, 0.1])
        if self.randomize_model_param:
            self.system.model_param = ModelParameter.create_random()
            print("reset with model_param: ", self.system.model_param)
        else:
            self.system.model_param = ModelParameter()
        return self.system.state, {}

    def step(
        self, action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.episode_count += 1
        self.system.step(action[0])
        obs = self.system.state
        is_uplight = self.system.is_uplight()
        if is_uplight:
            self.num_uplight += 1
            # print("success: ", self.num_uplight)
        reward = +1.0 if is_uplight else 0.0
        
        is_terminated = self.episode_count > 1000 and is_uplight
        return obs, reward, is_terminated, False, {}


class MyCallback(BaseCallback):
    def _on_rollout_start(self) -> None:
        vecenv: DummyVecEnv = self.training_env
        env: CartpoleEnv = vecenv.envs[0]
        env.num_uplight = 0

    def _on_step(self) -> bool:
        return True


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", help="train or test")
    args = parser.parse_args()

    model_name = "ppo_acrobot"
    if args.mode == "train":
        env = CartpoleEnv(randomize_model_param=True)
        # model = PPO(PartiallyAnalyticPolicy, env, verbose=1, n_steps=2048 * 2, tensorboard_log="./logs/")

        model = PPO(PartiallyAnalyticPolicy, env, verbose=1, n_steps=2048 * 2, 
                    policy_kwargs = dict(
                        activation_fn=th.nn.Tanh,
                        net_arch=dict(pi=[20, 20], vf=[20, 20]), ## changed from [256, 128]
                        log_std_init=-0.5,
                        ))
        callback = MyCallback()
        check_callback = CheckpointCallback(save_freq=2048 * 2 * 10, save_path="./models/", name_prefix=model_name)
        callback_list = CallbackList([callback, check_callback])
        model.learn(total_timesteps=2000000, callback=callback_list)
        model.save(model_name)
    else:
        env = CartpoleEnv(randomize_model_param=True)
        model = PPO.load("./models/ppo_acrobot_1966080_steps.zip")
                                   
        obs, _ = env.reset()
        for i in range(10000):
            print(i)
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            if done:
                print("done")
                break
