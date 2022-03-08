from moviepy.editor import ImageSequenceClip
import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointPosition
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import CloseBox
from rlbench.demo import Demo

from mohou.types import RGBImage


if __name__ == '__main__':
    obs_config = ObservationConfig()
    obs_config.set_all(True)

    env = Environment(
        action_mode=MoveArmThenGripper(
            arm_action_mode=JointPosition(), gripper_action_mode=Discrete()),
        obs_config=ObservationConfig(),
        headless=True)
    env.launch()

    task = env.get_task(CloseBox)
    task.reset()
    rgb_seq = []
    for i in range(30):
        gripper = 0.0
        action = np.ones(7) * 0.01 * i
        obs, _, _ = task.step(np.array(action.tolist() + [0]))
        rgb = RGBImage(obs.overhead_rgb)
        rgb_seq.append(rgb)

    clip = ImageSequenceClip([img.numpy() for img in rgb_seq], fps=50)
    clip.write_gif("tmp.gif", fps=50, loop=1)

