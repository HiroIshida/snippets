import time
import numpy as np
import torch
import torch.nn as nn
import flax.linen as fnn
import jax
import jax.numpy as jnp

layres = nn.Sequential(
        nn.Linear(40, 40),
        nn.ReLU(),
        nn.Linear(40, 40),
        nn.ReLU(),
        nn.Linear(40, 40),
        nn.ReLU(),
        nn.Linear(40, 1),
        nn.Sigmoid())


def _convert(torch_layer: nn.Module) -> fnn.Module:
    if isinstance(torch_layer, nn.Linear):
        feat = torch_layer.out_features
        return fnn.Dense(feat)
    elif isinstance(torch_layer, nn.ReLU):
        return fnn.relu
    elif isinstance(torch_layer, nn.Sigmoid):
        return fnn.sigmoid
    else:
        assert False


def convert(torch_seq: nn.Sequential) -> fnn.Sequential:
    layers = [_convert(tl) for tl in torch_seq]
    return fnn.Sequential(layers)


model = convert(layres)
batch = jnp.empty((100, 40))
variables = model.init(jax.random.PRNGKey(0), batch)


f = jax.jit(lambda p, x: model.apply(variables, batch))
f(variables, batch) # compile here

ts = time.time()
for _ in range(100):
    f(variables, batch)
perf = (time.time() - ts ) / 100.0
print("jax perf: {}".format(perf))

ts = time.time()
ten = torch.zeros(100, 40).float()
for _ in range(100):
    layres(ten)
perf = (time.time() - ts ) / 100.0
print("torch perf: {}".format(perf))
