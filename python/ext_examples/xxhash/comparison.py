import numpy as np
import xxhash
import hashlib
import time
a = np.random.randn(100, 10)

ts = time.time()
for _ in range(10000):
    h = xxhash.xxh64()
    h.update(a)
    val = h.intdigest()
    h.reset()
print(time.time() - ts) # 0.01

ts = time.time()
for _ in range(10000):
    hash(a.tostring())
print(time.time() - ts) # 0.03

ts = time.time()
for _ in range(10000):
    hashlib.md5(a.tostring()).digest()
print(time.time() - ts) # 0.1
