from functools import wraps
import functools
import hashlib
import operator
import xxhash
import time
import numpy as np


def array_cache(f):
    algo = lambda x: hashlib.md5(x).digest() # TODO? md5 is slow?
    cache = {}
    @wraps(f)
    def wrapped(*args, **kwargs):
        # NOTE __str__() is too big for nparray
        tobyte = lambda val: val.tostring() if type(val) is np.ndarray else val.__str__().encode()
        strenc = functools.reduce(operator.add, map(tobyte, args)) 
        if kwargs:
            strenc += functools.reduce(operator.add, map(tobyte, list(kwargs.values())))
        hashval = algo(strenc)
        if not hashval in cache:
            cache[hashval] = f(*args, **kwargs)
        return cache[hashval]
    return wrapped

@array_cache
def square(x, y=None, z=1):
    time.sleep(1)
    if y is None:
        return x @ x + z
    else:
        return y[0] + z

a = np.random.randn(100)
square(a, y=a)
square(a, y=a)
square(a)
square(a)
ts = time.time()
square(a, z=2)
print(time.time() -ts)
ts = time.time()
for i in range(10000):
    square(a, z=2)
print(time.time() -ts)
