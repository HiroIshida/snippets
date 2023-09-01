import tarfile
import pickle
import numpy as np
import tempfile
from pathlib import Path

with tarfile.open("tempdir.tar", "w") as tar:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        for i in range(100):
            arr = np.random.randn(1000)
            fp = td_path / f"arr{i}.pkl"
            with fp.open(mode = "wb") as f:
                pickle.dump(arr, f)
            tar.add(fp, arcname=fp.name)
