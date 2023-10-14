import numpy as np
import torch
from torch import Tensor
import torch.nn as nn


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        layers = []
        layers.append(nn.Linear(4, 12))
        layers.append(nn.Tanh())
        layers.append(nn.Linear(12, 12))
        layers.append(nn.Tanh())
        layers.append(nn.Linear(12, 1))
        self.layers = nn.Sequential(*layers)

    @property
    def dof(self) -> int:
        return sum([p.numel() for p in self.parameters()])

    def set_parameter(self, set_param: np.ndarray):
        assert len(set_param) == self.dof
        head = 0
        for name, param in self.named_parameters():
            n = np.prod(param.shape)
            tail = head + n
            param_partial = set_param[head:tail].reshape(tuple(param.shape))
            param.data = param.new_tensor(param_partial)
            head += n

    def get_parameter(self) -> np.ndarray:
        return np.concatenate([p.detach().cpu().numpy().flatten() for p in self.parameters()])

    def forward(self, x):
        return self.layers(x)

if __name__ == "__main__":
    net = Net()
    for _ in range(10):
        param = np.random.randn(net.dof)
        net.set_parameter(param)
        np.testing.assert_almost_equal(net.get_parameter(), param)
