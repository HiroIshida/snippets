import pickle
from pathlib import Path

p = Path("./data.pkl")
with p.open(mode = "rb") as f:
    a = pickle.load(f)

print(a)
