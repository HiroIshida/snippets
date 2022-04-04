import pickle
import numpy as np
from typing import List

np.random.seed(1)
data_list = [np.random.randn(s, s) for s in [2, 300]]

print("writing...")
with open('tmp', 'wb') as f:
    for data in data_list:
        byte = pickle.dumps(data)
        f.write(len(byte).to_bytes(8, 'big'))
        f.write(byte)
    f.write(int(0).to_bytes(8, 'big'))

print("reading...")
f = open('tmp', 'rb')
while True:
    size = int.from_bytes(f.read(8), 'big')
    if size == 0:
        break
    byte_object = f.read(size)
    data = pickle.loads(byte_object)
    print(data.shape)
