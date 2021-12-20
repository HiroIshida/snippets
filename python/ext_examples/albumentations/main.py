from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.datasets as datasets

import albumentations as A
from albumentations.pytorch import ToTensorV2

# load rgb image as numpy
im = Image.open("./sample.png")
rgb_im = np.array(im.convert('RGB'))

aug_composed = A.Compose([A.GaussNoise(p=1), A.RGBShift(r_shift_limit=40, g_shift_limit=40, b_shift_limit=40)])

fig = plt.figure()
for i in range(10):
    ax = fig.add_subplot(1, 10, i+1)
    ax.imshow(aug_composed(image=rgb_im)['image'])
plt.show()
