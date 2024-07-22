import pickle
import torch
import numpy as np
import tqdm
from mohou.types import (
    AngleVector,
    DepthImage,
    ElementSequence,
    EpisodeBundle,
    EpisodeData,
    GripperState,
    RGBImage,
)
from mohou.file import get_project_path
from build_data import DummyEpisode, convert_to_lerobot_dataset

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub.utils import (
    concatenate_episodes,
    save_images_concurrently,
)
from lerobot.common.datasets.utils import (
    calculate_episode_data_index,
    hf_transform_to_torch,
)
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

if __name__ == "__main__":
    pp = get_project_path("rlbench_demo_CloseDrawer")

    bundle = EpisodeBundle.load(pp)
    dummy_episode_list = []
    for episode in tqdm.tqdm(bundle):
        images = episode.get_sequence_by_type(RGBImage)
        images_np = np.array([e.numpy() for e in images])
        vecs_av = episode.get_sequence_by_type(AngleVector)
        vecs_av_np = np.array([e.numpy() for e in vecs_av])
        if GripperState in episode.type_shape_table:
            vecs_g = episode.get_sequence_by_type(GripperState)
            vecs_g_np = np.array([e.numpy() for e in vecs_g])
            vecs_np = np.concatenate([vecs_av_np, vecs_g_np], axis=1)
        else:
            vecs_np = vecs_av_np
        dummy_episode = DummyEpisode(images_np[1:], vecs_np[:-1], vecs_np[1:])
        dummy_episode_list.append(dummy_episode)

    dataset = convert_to_lerobot_dataset(dummy_episode_list, 10)
    # cache stats file
    with open("stats.pkl", "wb") as f:
        pickle.dump(dataset.stats, f)

    delta_timestamps = {
        "observation.images": [-0.1, 0.0],
        "observation.state": [-0.1, 0.0],
        "action": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
    }
    dataset.delta_timestamps = delta_timestamps
    print(f"finsh creating lerobot_dataset")

    training_steps = 10000
    device = torch.device("cuda")
    log_freq = 250

    cfg = DiffusionConfig()
    # cfg.input_shapes["observation.state"] = [8]
    # cfg.output_shapes["action"] = [8]
    policy = DiffusionPolicy(cfg, dataset_stats=dataset.stats)
    policy.train()
    policy.to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

    # Create dataloader for offline training.
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=64,
        shuffle=True,
        pin_memory=device != torch.device("cpu"),
        drop_last=True,
    )

    # Run training loop.
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
                torch.save(policy.state_dict(), "model.pth")
            step += 1
            if step >= training_steps:
                done = True
                break
    policy.save_pretrained(pp)
