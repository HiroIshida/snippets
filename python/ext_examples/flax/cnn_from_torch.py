import jax
import numpy as np
from dataclasses import dataclass
import argparse
import time
import torch.nn as nn
import torch
import torch.nn.functional as F
import onnxruntime as ort
import jax.numpy as jnp
from flax import linen as fnn
from typing import Dict, Tuple, Any

encoder_layers = [
    nn.Conv2d(1, 8, 3, padding=1, stride=(2, 2)),  # 14x14
    nn.ReLU(inplace=True),
    nn.Conv2d(8, 16, 3, padding=1, stride=(2, 2)),
    nn.ReLU(inplace=True),
    nn.Conv2d(16, 32, 3, padding=1, stride=(2, 2)),
    nn.ReLU(inplace=True),
    nn.Conv2d(32, 64, 3, padding=1, stride=(2, 2)),
    nn.ReLU(inplace=True),
    nn.Flatten(),
    # nn.Linear(1024, 1000),
    # nn.ReLU(inplace=True),
]
# encoder_layers = [
#     nn.Conv2d(1, 8, 3, stride=(2, 2)),  # 14x14
# ]

@dataclass
class FlaxConv:
    model: fnn.Conv
    variables: Dict

    def __call__(self, x) -> jnp.ndarray:
        return self.model.apply(self.variables, x)

    @classmethod
    def from_torch(cls, t_conv: nn.Conv2d) -> "FlaxConv":
        kernel = t_conv.weight.detach().cpu().numpy()
        bias = t_conv.bias.detach().cpu().numpy()
        features = t_conv.out_channels
        kernel_size = t_conv.kernel_size
        padding = t_conv.padding    

        kernel = jnp.transpose(kernel, (2, 3, 1, 0))
        variables = {'params': {'kernel': kernel, 'bias': bias}}
        j_conv = fnn.Conv(features=features, kernel_size=kernel_size, padding=padding)
        return cls(j_conv, variables)


@dataclass
class FlaxLinear:
    w: np.ndarray
    b: np.ndarray

    @classmethod
    def from_torch(cls, t_linear: nn.Linear):
        w = t_linear.weight.detach().cpu().numpy()
        b = t_linear.bias.detach().cpu().numpy()
        return cls(w, b)

    @classmethod
    def from_numpy(cls, w_: np.ndarray, b: np.ndarray):
        w = jax.numpy.transpose(w_, (1, 0))
        return cls(jnp.array(w), jnp.array(b))

    def __call__(self, x):
        return jnp.dot(x, self.w) + self.b



class FlaxFlatten:

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x.reshape((x.shape[0], -1))


def to_flax(module: nn.Module) -> Any:
    if isinstance(module, nn.Conv2d):
        return FlaxConv.from_torch(module)
    elif isinstance(module, nn.ReLU):
        return fnn.relu
    elif isinstance(module, nn.Flatten):
        return FlaxFlatten()
    elif isinstance(module, nn.Linear):
        return FlaxLinear.from_torch(module)
    else:
        assert False


flax_layers = []
for encoder in encoder_layers:
    flax_layers.append(to_flax(encoder))
model = fnn.Sequential(flax_layers)
dummy_input = jnp.ones((1, 56, 56, 1))  # note: order is different from pytorch
variables = model.init(jax.random.PRNGKey(0), dummy_input)
f = jax.jit(lambda x: model.apply(variables, x))
f(dummy_input)  # jit compile

print("start")
ts = time.time()
n = 1000
for _ in range(n):
    f(dummy_input)
print((time.time() - ts) / n)
