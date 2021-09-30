import skrobot
import time
from PIL import Image
import io
import numpy as np

robot = skrobot.models.PR2()
viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))
viewer.add(robot)
viewer.show()
time.sleep(1.0)
png_byte = viewer.scene.save_image((100, 100))
img = Image.open(io.BytesIO(png_byte))
arr = np.asarray(img)

import matplotlib.pyplot as plt
plt.imshow(arr)
plt.show()
