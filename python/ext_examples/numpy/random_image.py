import numpy as np
import matplotlib.pyplot as plt

img = np.random.randint(256, size=(10, 10, 3))
plt.imshow(img)
plt.show()
