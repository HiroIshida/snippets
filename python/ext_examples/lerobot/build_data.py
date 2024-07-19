import torch
import numpy as np
from dataclasses import dataclass
from typing import List
from PIL import Image as PILImage
from datasets import Dataset, Features, Image, Sequence, Value
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub.utils import (
    concatenate_episodes,
    save_images_concurrently,
)
from lerobot.common.datasets.utils import (
    calculate_episode_data_index,
    hf_transform_to_torch,
)

class DummyEpisode:
    images: np.ndarray
    states: np.ndarray
    actions: np.ndarray

    def __init__(self, T: int):
        image_list = []
        state_list = []
        action_list = []
        for _ in range(T):
            rand_image = np.random.randint(0, 256, size=(56, 56, 3), dtype=np.uint8)
            rand_state = np.random.rand(7)
            rand_action = np.random.rand(7)
            image_list.append(rand_image)
            state_list.append(rand_state)
            action_list.append(rand_action)
        self.images = np.array(image_list)
        self.states = np.array(state_list)
        self.actions = np.array(action_list)


def convert_to_data_dict(episodes: List[DummyEpisode], fps: int):
    # copied and tweaked from
    # https://github.com/ojh6404/imitator/blob/lerobot/imitator/scripts/lerobot_dataset_builder.py

    # create data dict
    ep_dicts = []
    for idx, ep in enumerate(episodes):
        T = len(ep.images)
        dones = torch.zeros(T, dtype=torch.bool)
        dones[-1] = True

        ep_dict = {}
        ep_dict["observation.images.camera"] = [PILImage.fromarray(im) for im in ep.images]
        ep_dict["observation.state"] = torch.from_numpy(ep.states).float()
        ep_dict["action"] = torch.from_numpy(ep.actions).float()
        ep_dict["episode_index"] = torch.ones(T, dtype=torch.int64) * idx
        ep_dict["frame_index"] = torch.arange(0, T, 1)
        ep_dict["timestamp"] = torch.arange(0, T, 1) / fps
        ep_dict["next.done"] = dones
        ep_dicts.append(ep_dict)
    data_dict = concatenate_episodes(ep_dicts)
    total_frames = data_dict["frame_index"].shape[0]
    data_dict["index"] = torch.arange(0, total_frames, 1)

    # create haggingface dataset
    features = Features(
        {
            "observation.images.camera": Image(),
            "observation.state": Sequence(Value("float32")),
            "action": Sequence(Value("float32")),
            "episode_index": Value("int64"),
            "frame_index": Value("int64"),
            "timestamp": Value("float32"),
            "next.done": Value("bool"),
            "index": Value("int64"),
        }
    )
    hf_dataset = Dataset.from_dict(data_dict, features=features)

    episode_data_index = calculate_episode_data_index(hf_dataset)
    info = {
        "fps": fps,
        "video": False,
    }
    lerobot_dataset = LeRobotDataset.from_preloaded(
        hf_dataset=hf_dataset,
        episode_data_index=episode_data_index,
        info=info,
    )
    return lerobot_dataset


if __name__ == "__main__":
    episode_list = []
    for _ in range(30):
        episode_list.append(DummyEpisode(80))
    hf_dataset = convert_to_data_dict(episode_list, 2)
    print(hf_dataset)
