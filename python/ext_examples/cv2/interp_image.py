import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d, NearestNDInterpolator

gray = np.zeros((100, 200)).astype(np.int16)
gray[30:60, 50:100] = +1
gray[50:80, 80:150] = -1
#gray_new = cv2.resize(gray, (100, 50), interpolation=cv2.INTER_NEAREST)
gray_new = cv2.resize(gray, (100, 50), interpolation=cv2.INTER_LINEAR)
print(set(gray_new.flatten().tolist()))
plt.imshow(gray)
plt.show()
