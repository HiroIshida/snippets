import time

from matplotlib import image
import numpy as np
import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

dummy_image = np.random.randint(0, high=255, size=(224, 224, 3))
aug_composed = A.Compose([A.GaussNoise(p=1), A.RGBShift(r_shift_limit=40, g_shift_limit=40, b_shift_limit=40)])

ts = time.time()
for i in tqdm.tqdm(range(3000)): # takes 20 sec ... too slow
    aug_composed(image=dummy_image)
print(time.time() - ts)
