import tqdm
import torch
import torch.nn as nn
from torch2jax import j2t, t2j
import jax
import jax.numpy as jnp
from flax import linen as fnn
from flax.core import freeze, unfreeze


class EncoderTorch(nn.Module):
    def __init__(self, n_channel: int, dim_bottleneck: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(n_channel, 8, 3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 16, dim_bottleneck),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float32

    # pytorch 
    torch_model = EncoderTorch(n_channel=1, dim_bottleneck=200).to(device, dtype)
    batch = torch.randn(1, 1, 122, 122, device=device, dtype=dtype)
    torch_output = torch_model(batch)

    traced = torch.jit.trace(torch_model.eval(), (batch,))
    torch_model_jit = torch.jit.optimize_for_inference(traced)

    for _ in range(10):
        torch_output = torch_model_jit(batch)
    for _ in tqdm.tqdm(range(30000)):
        torch_output = torch_model_jit(batch)

    # jax
    # jax_batch = jax.device_put(t2j(batch))
    # jax_model = t2j(torch_model)
    # params = {k: jax.device_put(t2j(v))
    #           for k, v in torch_model.named_parameters()}
    jax_batch = jax.device_put(t2j(batch))
    jax_model = t2j(torch_model)                               # モデル本体は stateless
    params = {k: jax.device_put(t2j(v))
              for k, v in torch_model.named_parameters()}

    jax_model_jit = jax.jit(jax_model)

    for _ in range(10):
        jax_output = jax_model_jit(jax_batch, state_dict=params)

    for _ in tqdm.tqdm(range(30000)):
        jax_output = jax_model_jit(jax_batch, state_dict=params)


