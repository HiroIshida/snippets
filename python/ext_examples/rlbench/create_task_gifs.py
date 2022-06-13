import argparse
import inspect
from pathlib import Path
from typing import Type

import numpy as np
import rlbench.tasks
import tqdm
from moviepy.editor import ImageSequenceClip
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.task import Task
from rlbench.demo import Demo
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig

from typing import List


def rlbench_demo_to_rgb_seq(demo: Demo) -> List[np.ndarray]:
    seq = []
    for obs in demo:
        rgbs = []
        for camera_name in ["overhead", "left_shoulder", "right_shoulder", "front", "wrist"]:
            rgbs.append(obs.__dict__[camera_name + "_rgb"])
        rgb = np.concatenate(rgbs, axis=1)
        seq.append(rgb)
    return seq


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-pn", type=str, default="rlbench_close_box", help="project name"
    )
    parser.add_argument("-tn", type=str, default="CloseDrawer", help="task name")
    parser.add_argument("-cn", type=str, default="overhead", help="camera name")
    parser.add_argument("-resol", type=int, default=112, help="epoch num")
    args = parser.parse_args()
    project_name = args.pn
    resolution = args.resol

    assert resolution in [112, 224]

    # Data generation by rlbench
    obs_config = ObservationConfig()
    obs_config.set_all(True)

    env = Environment(
        action_mode=MoveArmThenGripper(
            arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()
        ),
        obs_config=ObservationConfig(),
        headless=True,
    )
    env.launch()

    all_task_classes = []
    for attr in rlbench.tasks.__dict__.values():
        if not inspect.isclass(attr):
            continue
        if issubclass(attr, Task):
            all_task_classes.append(attr)

    dirpath = Path("./rlbench_gif")
    dirpath.mkdir(exist_ok=True)

    for task_class in all_task_classes:
        print("executing task class named {}".format(task_class.__name__))
        task = env.get_task(task_class)

        mohou_episode_data_list = []

        demo = task.get_demos(amount=1, live_demos=True)[0]
        rgb_seq = rlbench_demo_to_rgb_seq(demo)
        clip = ImageSequenceClip([img for img in rgb_seq], fps=50)
        file_path = dirpath / (task_class.__name__ + ".gif")
        clip.write_gif(str(file_path), fps=50)
