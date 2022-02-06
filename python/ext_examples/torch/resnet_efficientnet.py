import contextlib
import numpy as np
import torch
import time
from torchvision.models import efficientnet_b0
from torchvision.models import resnet50
import torch.nn as nn

# input 224 x 224
# output (1000,)

@contextlib.contextmanager
def measure_time():
    ts = time.time()
    yield
    print(time.time() - ts)

def two_channel_efficientnetb0():
    model = efficientnet_b0()
    model.features[0][0] = nn.Conv2d(2, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    return model

def two_channel_resnet50():
    model = resnet50()
    model.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    return model


image = torch.from_numpy(np.random.randn(1, 2, 224, 224)).float()

for model in [two_channel_efficientnetb0(), two_channel_resnet50()]:
    with measure_time():
        out = model(image)
        print(out.shape)
