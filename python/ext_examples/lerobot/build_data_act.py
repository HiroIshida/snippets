import uuid
from pathlib import Path
import cv2
import numpy as np
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
import torch
import torchvision

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.act.modeling_act import ACTConfig, ACTPolicy
from lerobot.common.datasets.compute_stats import compute_stats
from build_data import convert_to_lerobot_dataset

@dataclass
class DummyEpisode:
    images: np.ndarray
    states: np.ndarray
    actions: np.ndarray

    @classmethod
    def create(cls, T: int):
        image_list = []
        state_list = []
        action_list = []
        for _ in range(T):
            rand_image = np.random.randint(0, 256, size=(96, 96, 3), dtype=np.uint8)
            rand_state = np.random.rand(8)
            rand_action = np.random.rand(8)
            image_list.append(rand_image)
            state_list.append(rand_state)
            action_list.append(rand_action)
        images = np.array(image_list)
        states = np.array(state_list)
        actions = np.array(action_list)
        return cls(images, states, actions)


if __name__ == "__main__":
    episode_list = []
    for _ in range(30):
        episode_list.append(DummyEpisode.create(200))
    dataset = convert_to_lerobot_dataset(episode_list, 10)
    delta_timestamps = {
        "action": [0.1 * i for i in range(100)],
    }
    dataset.delta_timestamps = delta_timestamps

    training_steps = 5000
    device = torch.device("cuda")
    log_freq = 250

    cfg = ACTConfig()
    cfg.input_shapes = {
        "observation.images": [3, 96, 96],
        "observation.state": [8],
    }
    cfg.output_shapes = {
        "action": [8],
    }
    cfg.input_normalization_modes = {
      "observation.images": "mean_std",
      "observation.state": "mean_std",
    }

    policy = ACTPolicy(cfg, dataset.stats)

    policy.train()
    policy.to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=64,
        shuffle=True,
        pin_memory=device != torch.device("cpu"),
        drop_last=True,
    )

    step = 0
    done = False
    while not done:
        for batch in dataloader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            output_dict = policy.forward(batch)
            loss = output_dict["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                print(f"step: {step} loss: {loss.item():.3f}")
            step += 1
            if step >= training_steps:
                done = True
                break
