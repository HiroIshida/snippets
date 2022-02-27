import time
import os
import tempfile
import h5py
import pickle
import codecs
from typing import Any, List

import numpy as np

serialize = lambda x: pickle.dumps(x)
n_chunk = 300
lst = [serialize(np.random.randn(1000, 1000)) for _ in range(n_chunk)]

with tempfile.TemporaryDirectory() as td:
    filename = os.path.join(td, 'hoge.h5')
    f = h5py.File(filename, 'w')
    f.create_dataset('lst', data=lst, chunks=(n_chunk,))
    f.close()

    ts = time.time()
    f = h5py.File(filename, 'r')
    a = f['lst'][1:10]
    f.close()
    print(time.time() - ts) # 0.02

    filename = os.path.join(td, 'hoge.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(lst, f)

    ts = time.time()
    with open(filename, 'rb') as f:
        b = pickle.load(f)
    print(time.time() - ts) # 0.83
