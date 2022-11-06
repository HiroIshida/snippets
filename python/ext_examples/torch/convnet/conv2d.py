import torch
import torch.nn as nn
from torch import Tensor

# use simple cnn
n_channel = 1
encoder_layers = [
    nn.Conv2d(n_channel, 8, 3, padding=1, stride=(2, 2)),  # 14x14
    nn.ReLU(inplace=True),
    nn.Conv2d(8, 16, 3, padding=1, stride=(2, 2)),
    nn.ReLU(inplace=True),
    nn.Conv2d(16, 32, 3, padding=1, stride=(2, 2)),
    nn.ReLU(inplace=True),
    nn.Conv2d(32, 64, 3, padding=1, stride=(2, 2)),
    nn.Flatten(),
    nn.Linear(1024, 500),
    nn.ReLU(inplace=True),
]

sample = torch.randn(10, 1, 56, 56)
for i in range(len(encoder_layers)):
    seq = nn.Sequential(*encoder_layers[:i+1])
    out = seq(sample)
    print(out.shape)

