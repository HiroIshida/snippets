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

aug_lst = []
aug_lst.append(A.Compose([A.GaussNoise(p=1)]))
aug_lst.append(A.Compose([A.RGBShift(p=1)]))
aug_lst.append(A.Compose([A.GaussNoise(p=1), A.RGBShift(p=1)]))

auged = [a(image=rgb_im) for a in aug_lst]

fig = plt.figure()
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)

ax1.imshow(auged[0]['image'])
ax2.imshow(auged[1]['image'])
ax3.imshow(auged[2]['image'])
plt.show()
