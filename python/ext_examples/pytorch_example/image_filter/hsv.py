from math import remainder
import torch
from torch.optim import Adam
from torch import Tensor
import torch.nn as nn
from mohou.types import RGBImage

class RGB2HSV(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, rgb_image: Tensor):
        n_sample, n_channel, nx, ny = rgb_image.shape
        r = rgb_image[:, 0, :, :]
        g = rgb_image[:, 1, :, :]
        b = rgb_image[:, 2, :, :]

        cmax = torch.max(r, torch.max(g, b))
        cmin = torch.min(r, torch.min(g, b))
        diff = cmax - cmin

        h = torch.zeros([n_sample, nx, ny])

        h += torch.where(
                torch.logical_and(r > g, r > b),
                torch.remainder((60 * ((g - b) / diff) + 360), 360),
                torch.zeros([n_sample, nx, ny]))

        h += torch.where(
                torch.logical_and(g > r, g > b),
                torch.remainder((60 * ((b - r) / diff) + 120), 360),
                torch.zeros([n_sample, nx, ny]))

        h += torch.where(
                torch.logical_and(b > r, b > g),
                torch.remainder((60 * ((r - g) / diff) + 240), 360),
                torch.zeros([n_sample, nx, ny]))

        return h


class HSVFilter(nn.Module):
    h_min: float
    h_max: float
    s_min: float
    s_max: float
    v_min: float
    v_max: float

    def __init__(self):
        super().__init__()


def rgb2hsv(inputt, epsilon=1e-10):
    assert(inputt.shape[1] == 3)

    r, g, b = inputt[:, 0], inputt[:, 1], inputt[:, 2]
    max_rgb, argmax_rgb = inputt.max(1)
    min_rgb, argmin_rgb = inputt.min(1)

    max_min = max_rgb - min_rgb + epsilon

    h1 = 60.0 * (g - r) / max_min + 60.0
    h2 = 60.0 * (b - g) / max_min + 180.0
    h3 = 60.0 * (r - b) / max_min + 300.0

    h = torch.stack((h2, h3, h1), dim=0).gather(dim=0, index=argmin_rgb.unsqueeze(0)).squeeze(0)
    s = max_min / (max_rgb + epsilon)
    v = max_rgb

    return torch.stack((h, s, v), dim=1)



if __name__ == '__main__':
    rgb = RGBImage.dummy_from_shape([100, 100])
    tmp = torch.unsqueeze(rgb.to_tensor(), dim=0)
    param = nn.Parameter(torch.tensor(tmp, requires_grad=True))

    optimizer = Adam([param], lr=0.1)

    #rgb2hsv = RGB2HSV()
    rgb_seq = [RGBImage.from_tensor(torch.squeeze(param.detach().clone(), dim=0))]
    for _ in range(2000):
        optimizer.zero_grad()
        h = rgb2hsv(param)[:, 2]
        print(h)
        loss_value = nn.MSELoss()(h, 0.5 * torch.ones(1, 100, 100))
        loss_value.backward()
        print(loss_value)
        optimizer.step()
        rgb_seq.append(RGBImage.from_tensor(torch.squeeze(param.detach().clone(), dim=0)))

    for rgb in rgb_seq:
        print(rgb.shape)

    from moviepy.editor import ImageSequenceClip
    clip = ImageSequenceClip([rgb.numpy() for rgb in rgb_seq], fps=50)
    clip.write_gif("unchi.gif", fps=50)
