# https://stackoverflow.com/questions/30730983/make-lru-cache-ignore-some-of-the-function-arguments

import numpy as np
from cachetools import cached
from cachetools.keys import hashkey
from mohou.types import RGBImage
import hashlib
import pickle

def pickle_hash(arr):
    a = hashlib.md5()
    a.update(pickle.dumps(arr))
    value = a.hexdigest()
    return value

@cached(cache={}, key=lambda obj: pickle_hash(obj))
def temp(image: RGBImage):
    print("internal computation")

img1 = RGBImage.dummy_from_shape((10, 10))
img2 = RGBImage.dummy_from_shape((10, 10))
temp(img1)
for i in range(100):
    temp(img1)
temp(img2)
for i in range(100):
    temp(img2)
