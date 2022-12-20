from dataclasses import dataclass
import time
import numpy as np
import torch
import torch.nn as nn
import flax.linen as fnn
import jax
import jax.numpy as jnp

layers = nn.Sequential(
        nn.Linear(40, 40),
        nn.ReLU(),
        nn.Linear(40, 40),
        nn.ReLU(),
        nn.Linear(40, 40),
        nn.ReLU(),
        nn.Linear(40, 1),
        nn.Sigmoid())


@dataclass
class JaxLinear:
    w: np.array
    b: np.array

    @classmethod
    def from_torch(cls, layer: nn.Linear):
        w_ = layer.weight.detach().numpy()
        w = jax.numpy.transpose(w_, (1, 0))
        b = layer.bias.detach().numpy() 
        return cls(jnp.array(w), jnp.array(b))

    def __call__(self, x):
        return jnp.dot(x, self.w) + self.b


def _convert(torch_layer: nn.Module) -> fnn.Module:
    if isinstance(torch_layer, nn.Linear):
        feat = torch_layer.out_features
        layer = JaxLinear.from_torch(torch_layer)
        return layer
    elif isinstance(torch_layer, nn.ReLU):
        return fnn.relu
    elif isinstance(torch_layer, nn.Sigmoid):
        return fnn.sigmoid
    else:
        assert False


def convert(torch_seq: nn.Sequential) -> fnn.Sequential:
    torch_first_layer: nn.Linear = torch_seq[0]
    layers = [_convert(tl) for tl in torch_seq]
    model = fnn.Sequential(layers)

    # define
    dummy_input = jnp.empty((torch_first_layer.in_features,))
    variables = model.init(jax.random.PRNGKey(0), dummy_input)
    f = jax.jit(lambda x: model.apply(variables, x).reshape(()))
    grad = jax.jit(jax.grad(f))

    # jit compile
    f(dummy_input)
    grad(dummy_input)
    return f, grad


def torch_eval(x: torch.Tensor):
    x.requires_grad_(True)
    y = layers(x)
    y.backward()
    return y.detach().numpy().item(), x.grad.detach().numpy()


def test_consistency():
    batch = np.random.randn(40)
    f_torch, g_torch = torch_eval(torch.from_numpy(batch).float())

    func_jax, grad_jax = convert(layers)
    f_jax, g_jax = func_jax(batch), grad_jax(batch)

    np.testing.assert_almost_equal(g_torch, g_jax)


def test_performace():
    batch = np.random.randn(40)
    n = 100
    ts = time.time()
    for _ in range(n):
        f_torch, g_torch = torch_eval(torch.from_numpy(batch).float())
    torch_score = (time.time() - ts) / n

    func_jax, grad_jax = convert(layers)
    ts = time.time()
    for _ in range(n):
        f_jax, g_jax = func_jax(batch), grad_jax(batch)
    jax_score = (time.time() - ts) / n

    speedup_x = torch_score / jax_score
    print("{}x speedup".format(speedup_x))
    assert speedup_x > 5


test_consistency()
test_performace()
