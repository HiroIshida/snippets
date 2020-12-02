import time 
import numpy as np
A = np.random.randn(200)
B = np.random.randn(10)

ts = time.time()
for i in range(1000):
    A - np.repeat(B, 20)
print(time.time() - ts)

ts = time.time()
for i in range(1000):
    (A.reshape(20, 10) - B[None, :]).reshape(200)
print(time.time() - ts)
