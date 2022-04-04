import pickle
import numpy as np
from typing import List

np.random.seed(1)
data_list = [np.random.randn(s, s) for s in [1, 2, 3, 4]]

def dump_list(lst, fname):
    with open(fname, 'wb') as f:
        location: int = 0
        for e in lst:
            byte = pickle.dumps(e)
            location += len(byte)
            f.write(location.to_bytes(8, 'big'))
        f.write(int(0).to_bytes(8, 'big'))

        for e in lst:
            byte = pickle.dumps(e)
            f.write(byte)

def load_list(fname, idxes):
    with open(fname, 'rb') as f:
        start_positions = [0]
        end_positions = []
        while True:
            end_pos = int.from_bytes(f.read(8), 'big')
            if end_pos == 0:
                start_positions.pop()
                break
            end_positions.append(end_pos)
            start_positions.append(end_pos + 1)

        offset = f.tell() + 1
        for idx in idxes:
            f.seek(offset + start_positions[idx])
            byte_object = f.read(end_positions[idx] - start_positions[idx] + 1)
            data = pickle.loads(byte_object)

dump_list(data_list, 'tmp2')
load_list('tmp2', [2, 3])
