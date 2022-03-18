import time
import tqdm
import os
import tempfile
import h5py
import pickle
import codecs
from typing import Any, List

import numpy as np

def test_chunk(n_chunk):
    serialize = lambda x: pickle.dumps(x)
    lst = [serialize(np.random.randn(1000, 1000)) for _ in tqdm.tqdm(range(n_chunk))]

    with tempfile.TemporaryDirectory() as td:
        filename = os.path.join(td, 'hoge.h5')
        f = h5py.File(filename, 'w')
        f.create_dataset('lst', data=lst, chunks=(n_chunk,))
        f.close()

n_chunk_under_4gb = 500
n_chunk_over_4gb = 550
print("testing under 4GB")
test_chunk(n_chunk_under_4gb)  # ok
print("testing over 4GB")
test_chunk(n_chunk_over_4gb)  # ng

