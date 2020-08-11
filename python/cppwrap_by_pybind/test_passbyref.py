import example;
import numpy as np
import time 
a = np.ones(20 * 7 * 3 * 7)

ts = time.time()
for i in range(10000):
    example.fbz_ref(a)
print("pass by ref : {0}".format(time.time() - ts))

ts = time.time()
for i in range(10000):
    b = example.fbz_val(a)
print("return by val  : {0}".format(time.time() - ts))

