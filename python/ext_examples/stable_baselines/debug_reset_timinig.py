import numpy as np
import argparse
import gymnasium as gym
from gymnasium.envs.classic_control.acrobot import AcrobotEnv
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv


class MyCallback(BaseCallback):
    def _on_rollout_start(self) -> None:
        print("rollout start")
        vecenv: DummyVecEnv = self.training_env
        env = vecenv.envs[0]
        print(env.state)

    def _on_step(self) -> bool:
        return True

state_list = []
reset_called = 0

class AcrobotMyEnv(AcrobotEnv):
    def step(self, a):
        state_list.append(self.state)
        if self._terminal():
            print("terminal")
        return super().step(a)

    def reset(self, *args, **kwargs):
        global reset_called
        reset_called += 1
        print("reset env")
        if reset_called > 1:
            assert False
        return super().reset(*args, **kwargs)


# argparse select mode
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="test", help="train or test")
args = parser.parse_args()

model_name = "ppo_acrobot"
if args.mode == "train":
    env = AcrobotMyEnv(render_mode=None)
    model = PPO(MlpPolicy, env, verbose=1)
    callback = MyCallback()
    model.learn(total_timesteps=10000, callback=callback)
    model.save(model_name)

    import matplotlib.pyplot as plt
    S = np.array(state_list)
    plt.plot(S)
    plt.show()
else:
    env = AcrobotEnv(render_mode="human")
    model = PPO.load(model_name)
    obs, _ = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            assert False
    env.close()
