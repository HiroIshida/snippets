import numpy as np
import argparse
import gymnasium as gym
from gymnasium.envs.classic_control.acrobot import AcrobotEnv
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from math import cos

state_list = []
reset_count = 0

class AcrobotMyEnv(AcrobotEnv):
    vel_limit: float

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.vel_limit = 1.0

    def step(self, a):
        state_list.append(self.state)
        ret = super().step(a)
        reward = ret[2]
        return ret

    def reset(self, *args, **kwargs):
        global reset_count
        reset_count += 1
        print("reset count: ", reset_count)
        return super().reset(*args, **kwargs)

    def _terminal(self):
        s = self.state
        assert s is not None, "Call reset before using AcrobotEnv object."
        return bool(-cos(s[0]) - cos(s[1] + s[0]) > 1.0 and abs(s[2]) < self.vel_limit and abs(s[3]) < self.vel_limit)

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="test", help="train or test")
args = parser.parse_args()

model_name = "ppo_acrobot"
if args.mode == "train":
    env = AcrobotMyEnv(render_mode=None)
    model = PPO(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=20000)
    reset_count = 0
    env.vel_limit = 0.8
    model.learn(total_timesteps=20000)
    reset_count = 0
    env.vel_limit = 0.6
    model.learn(total_timesteps=20000)
    reset_count = 0
    env.vel_limit = 0.4
    model.learn(total_timesteps=20000)
    # model.save(model_name)
else:
    env = AcrobotMyEnv(render_mode="human")
    env.vel_limit = 0.1
    model = PPO.load(model_name)
    obs, _ = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            assert False
    env.close()
