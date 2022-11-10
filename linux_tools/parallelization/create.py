import pickle
import shutil
from pathlib import Path
import numpy as np
import uuid

path = Path("./tmp")
shutil.rmtree(path, ignore_errors=True)
path.mkdir(exist_ok=True)

size = (56, 56, 28)

for _ in range(2000):
    a = np.random.randn(*size)
    fp = path / "{}.pkl".format(str(uuid.uuid4()))
    with fp.open(mode = "wb") as f:
        pickle.dump(a, f)
