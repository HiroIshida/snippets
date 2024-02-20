import torch
import torch.nn as nn
from torch.nn import ConvTranspose2d
from torch.nn import ReLU


class Fuser(nn.Module):
    def __init__(self, dim_vector: int):
        # with batch norm
        super(Fuser, self).__init__()
        conv_layers1 = nn.Sequential(*[
            nn.Conv2d(1, 8, 3, padding=1, stride=(2, 2)),  # 28 x 28
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        ])

        fusing_layers = nn.Sequential(*[
                nn.Linear(dim_vector, 28*28),
                nn.BatchNorm1d(28*28),
                nn.ReLU(inplace=True),
        ])

        conv_layer2 = nn.Sequential(*[
            nn.Conv2d(9, 16, 3, padding=1, stride=(2, 2)),  # 14 x 14
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1, stride=(2, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, stride=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),  # 64 x 4 x 4
        ])

        linear_layers = nn.Sequential(*[
            nn.Flatten(),
            nn.Linear(64*4*4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 1),
        ])

        self.conv_layers1 = conv_layers1
        self.fusing_layers = fusing_layers  
        self.conv_layers2 = conv_layer2
        self.linear_layers = linear_layers

    def forward(self, image, vector):
        image = self.conv_layers1(image)
        imaged_vector = self.fusing_layers(vector)
        imaged_vector = imaged_vector.view(-1, 1, 28, 28)
        image_fused = torch.cat((image, imaged_vector), 1)
        out = self.linear_layers(self.conv_layers2(image_fused))
        return out


# 56x56 
sample_images = torch.randn(1, 1, 56, 56)
vector = torch.randn(1, 10)
fuser = Fuser(10)
# eval mode
fuser.eval()
output = fuser(sample_images, vector)
print(f"Output shape: {output.shape}")
