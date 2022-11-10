import pickle
from pathlib import Path
import numpy as np
import uuid

for _ in range(10000):
    a = np.random.randn(50, 50, 20)
    fn = Path("./tmp/{}.pkl".format(str(uuid.uuid4())))
    with fn.open(mode = "wb") as f:
        pickle.dump(a, f)
