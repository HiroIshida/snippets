import cv2
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

img = cv2.imread("./sample.png")
tensor = torchvision.transforms.ToTensor()(img)
img_rev = torchvision.transforms.ToPILImage()(tensor)
np_array = np.asarray(img_rev)
print(np_array)
