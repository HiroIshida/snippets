import torch
import torch.nn as nn

n_channel = 1
data = torch.randn(10, n_channel, 56, 56, 28)

channel_list = [8, 16]

n_conv_out_dim = 1000


encoder_layers = [
    nn.Conv3d(n_channel, 8, (3, 3, 2), padding=1, stride=(2, 2, 1)),
    nn.Conv3d(8, 16, (3, 3, 3), padding=1, stride=(2, 2, 2)),
    nn.Conv3d(16, 32, (3, 3, 3), padding=1, stride=(2, 2, 2)),
    nn.Conv3d(32, 64, (3, 3, 3), padding=1, stride=(2, 2, 2)),
]

decoder_layers = [
    nn.ConvTranspose3d(64, 32, 3, padding=1, stride=2),
    nn.ConvTranspose3d(32, 16, 4, stride=2, padding=1),
    nn.ConvTranspose3d(16, 8, 4, stride=2, padding=1),
    nn.ConvTranspose3d(8, 1, (4, 4, 3), stride=(2, 2, 1), padding=(1, 1, 1))
]

for i in range(len(encoder_layers)):
    seq = nn.Sequential(*encoder_layers[:i+1])
    out = seq(data)
    print(out.shape)

data = out
for i in range(len(decoder_layers)):
    seq = nn.Sequential(*decoder_layers[:i+1])
    out = seq(data)
    print(out.shape)
