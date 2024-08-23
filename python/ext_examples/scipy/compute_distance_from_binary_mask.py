import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

size = 100
center = (size // 2, size // 2)
radius = 20

Y, X = np.ogrid[:size, :size]
dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
binary_mask = dist_from_center <= radius

distance_matrix = ndimage.distance_transform_edt(~binary_mask)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Binary Mask")
plt.imshow(binary_mask, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Distance Matrix")
plt.imshow(distance_matrix, cmap='viridis')
plt.colorbar()

plt.show()
