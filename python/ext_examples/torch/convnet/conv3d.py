import torch
import torch.nn as nn

n_channel = 1
data = torch.randn(10, n_channel, 56, 56, 28)

channel_list = [8, 16]

encoder_layers = [
    nn.Conv3d(n_channel, 8, (3, 3, 2), padding=1, stride=(2, 2, 1)),
    nn.ReLU(inplace=True),
    nn.Conv3d(8, 16, (3, 3, 3), padding=1, stride=(2, 2, 2)),
    nn.ReLU(inplace=True),
    nn.Conv3d(16, 32, (3, 3, 3), padding=1, stride=(2, 2, 2)),
    nn.ReLU(inplace=True),
    nn.Conv3d(32, 64, (3, 3, 3), padding=1, stride=(2, 2, 2)),
    nn.Flatten(),
    nn.Linear(4096, 800),
    nn.ReLU(inplace=True),
]

for i in range(len(encoder_layers)):
    seq = nn.Sequential(*encoder_layers[:i+1])
    out = seq(data)
    print(out.shape)

