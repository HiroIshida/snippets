import contextlib
import numpy as np

global vec
vec = np.zeros(3)

@contextlib.contextmanager
def mymanager(slide: np.ndarray):
    global vec
    vec += slide
    yield
    vec -= slide

with mymanager(np.ones(3)):
    print(vec)
print(vec)



