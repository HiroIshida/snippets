from dummy import Example
import pickle
from pathlib import Path

a = Example(1, 1.0)
p = Path("./data.pkl")
with p.open(mode = "wb") as f:
    pickle.dump(a, f)
