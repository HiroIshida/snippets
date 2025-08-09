import torch
import time
import pickle
import argparse
from typing import Type
from pathlib import Path

import numpy as np
import rlbench.tasks
import tqdm
from moviepy.editor import ImageSequenceClip
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointPosition
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.observation import Observation
from rlbench.backend.task import Task
from rlbench.environment import Environment

from mohou.file import get_project_path
from mohou.propagator import LSTMPropagator
from mohou.types import (
    AngleVector,
    DepthImage,
    ElementDict,
    EpisodeBundle,
    GripperState,
    RGBImage,
)

from rlbench.observation_config import CameraConfig, ObservationConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy


def setup_observation_config(camera_name: str, resolution: int) -> ObservationConfig:
    camera_names = {"left_shoulder", "right_shoulder", "overhead", "wrist", "front"}
    assert camera_name in camera_names

    kwargs = {}
    ignore_camera_names = camera_names.difference(camera_name)
    for ignore_name in ignore_camera_names:
        kwargs[ignore_name + "_camera"] = CameraConfig(
            rgb=False, depth=False, point_cloud=False, mask=False
        )

    kwargs[camera_name + "_camera"] = CameraConfig(
        image_size=(resolution, resolution), point_cloud=False, mask=False
    )
    return ObservationConfig(**kwargs)



def edict_to_action(edict: ElementDict) -> np.ndarray:
    av_next = edict[AngleVector]
    gs_next = edict[GripperState]
    return np.hstack([av_next.numpy(), gs_next.numpy()])


def obs_to_edict(obs: Observation, resolution: int, camera_name: str) -> ElementDict:
    av = AngleVector(obs.joint_positions)
    gs = GripperState(np.array([obs.gripper_open]))

    arr_rgb = obs.__dict__[camera_name + "_rgb"]
    arr_depth = np.expand_dims(obs.__dict__[camera_name + "_depth"], axis=2)

    rgb = RGBImage(arr_rgb)
    depth = DepthImage(arr_depth)
    return ElementDict([av, gs, rgb, depth])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pn", type=str, default="rlbench_demo_CloseDrawer", help="project name")
    parser.add_argument("-tn", type=str, default="CloseDrawer", help="task name")
    parser.add_argument("-cn", type=str, default="overhead", help="camera name")
    parser.add_argument("-n", type=int, default=250, help="step num")
    parser.add_argument("-m", type=int, default=3, help="simulation num")
    args = parser.parse_args()
    project_name: str = args.pn
    task_name: str = args.tn
    camera_name: str = args.cn
    n_step: int = args.n
    n_sim: int = args.m

    project_path = get_project_path(project_name)

    bundle = EpisodeBundle.load(project_path)
    resolution = bundle.spec.type_shape_table[RGBImage][0]

    obs_config = setup_observation_config(camera_name, resolution)
    env = Environment(
        action_mode=MoveArmThenGripper(
            arm_action_mode=JointPosition(), gripper_action_mode=Discrete()
        ),
        obs_config=obs_config,
        headless=True,
    )
    env.launch()

    untouch_bundle = bundle.get_untouch_bundle()
    av_init = untouch_bundle[0].get_sequence_by_type(AngleVector)[0]
    gs_init = untouch_bundle[0].get_sequence_by_type(GripperState)[0]
    edict_init = ElementDict([av_init, gs_init])

    assert hasattr(rlbench.tasks, task_name)
    task_type: Type[Task] = getattr(rlbench.tasks, task_name)
    task = env.get_task(task_type)

    # device = "cuda"
    device = "cuda"

    for i in range(n_sim):
        task.reset()

        # load ./model.pth pytorch
        with Path("./stats.pkl").open("rb") as f:
            stats = pickle.load(f)
        for key, value in stats.items():
            for key_sub, value_sub in value.items():
                stats[key][key_sub] = value_sub.to(device)
        policy = DiffusionPolicy(dataset_stats=stats)
        policy = policy.to(device)
        policy.load_state_dict(torch.load("./model.pth"))

        rgb_seq_gif = []

        obs, _, _ = task.step(edict_to_action(edict_init))
        edict = obs_to_edict(obs, resolution, camera_name)
        for _ in tqdm.tqdm(range(n_step)):
            rgb = edict[RGBImage]
            image = torch.from_numpy(rgb.numpy()).float()
            image = image.to(torch.float32) / 255
            image = image.permute(2, 0, 1)
            image = image.unsqueeze(0)

            pre_action = edict_to_action(edict)
            state = torch.from_numpy(pre_action).float().unsqueeze(0)
            image = image.to(device)
            state = state.to(device)

            observation = {
                "observation.images": image,
                "observation.state": state
            }
            ts = time.time()
            with torch.inference_mode():
                action = policy.select_action(observation)
            print(f"time to infer {time.time() - ts}")
            action = action.detach().cpu().numpy().flatten()
            obs, _, _ = task.step(action)
            edict = obs_to_edict(obs, resolution, camera_name)

            rgb_seq_gif.append(RGBImage(obs.__dict__[camera_name + "_rgb"]))

        file_path = project_path / "feedback_simulation-{}.gif".format(i)
        clip = ImageSequenceClip([img.numpy() for img in rgb_seq_gif], fps=50)
        clip.write_gif(str(file_path), fps=50)
