import torch
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy, DiffusionConfig
import pickle

torch.manual_seed(0)

resol = 56
camera_names = ["gripper", "webcam"]
input_shapes = {"observation.state": [6]}
for name in camera_names:
    input_shapes[f"observation.image.{name}"] = [3, resol, resol]
output_shapes = {"action": [6]}

# determine normalizer
normalization_mode = {"observation.state": "min_max"}
for name in camera_names:
    normalization_mode[f"observation.image.{name}"] = "mean_std"

# create policy
with open("stats-diffusion.pkl", "rb") as f:
    stats = pickle.load(f)
conf = DiffusionConfig(input_shapes=input_shapes, output_shapes=output_shapes, input_normalization_modes=normalization_mode, crop_shape=None, num_inference_steps=5)
policy = DiffusionPolicy(conf, dataset_stats=stats)

# inference
observation = {"observation.image.gripper": torch.rand(1, 3, resol, resol),
               "observation.image.webcam": torch.rand(1, 3, resol, resol),
               "observation.state": torch.rand(1, 6)}

print("start inference")
action = policy.select_action(observation)
print(action)
