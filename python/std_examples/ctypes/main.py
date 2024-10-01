import tqdm
import matplotlib.pyplot as plt
import ctypes
import numpy as np
from typing import List

# Load the shared library
lib = ctypes.CDLL('./lib.so')  # Adjust the path as needed

# Define the argument and return types for each function
lib.make_boxes.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int
]
lib.make_boxes.restype = ctypes.c_void_p

lib.delete_boxes.argtypes = [ctypes.c_void_p]
lib.delete_boxes.restype = None

lib.signed_distance.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_void_p]
lib.signed_distance.restype = ctypes.c_double

lib.signed_distance_batch.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_void_p
]
lib.signed_distance_batch.restype = None

class Boxes:
    def __init__(self, xmin: List[float], xmax: List[float], ymin: List[float], ymax: List[float]):
        n = len(xmin)
        self.ptr = lib.make_boxes(
            (ctypes.c_double * n)(*xmin),
            (ctypes.c_double * n)(*xmax),
            (ctypes.c_double * n)(*ymin),
            (ctypes.c_double * n)(*ymax),
            ctypes.c_int(n)
        )

    def __del__(self):
        lib.delete_boxes(self.ptr)

    def signed_distance(self, x: float, y: float) -> float:
        return lib.signed_distance(ctypes.c_double(x), ctypes.c_double(y), self.ptr)

    def signed_distance_batch(self, x: List[float], y: List[float]) -> List[float]:
        n = len(x)
        dist = (ctypes.c_double * n)()
        lib.signed_distance_batch(
            (ctypes.c_double * n)(*x),
            (ctypes.c_double * n)(*y),
            dist,
            ctypes.c_int(n),
            self.ptr
        )
        return list(dist)


if __name__ == "__main__":
    xmins = [0.0, 2.0, 4.0]
    xmaxs = [1.0, 3.0, 5.0]
    ymins = [0.0, 2.0, 4.0]
    ymaxs = [1.0, 3.0, 5.0]

    for _ in tqdm.tqdm(range(10000000)):
        boxes = Boxes(xmins, xmaxs, ymins, ymaxs)

    import time
    time.sleep(100)
    visualise = False
    if visualise:
        xmin = -0.5
        ymin = -0.5
        xmax = 6.5
        ymax = 6.5
        xlin = np.linspace(xmin, xmax, 100)
        ylin = np.linspace(ymin, ymax, 100)
        X, Y = np.meshgrid(xlin, ylin)
        pts = np.vstack([X.ravel(), Y.ravel()]).T
        dist = boxes.signed_distance_batch(pts[:, 0], pts[:, 1])
        dist = np.array(dist).reshape(X.shape)

        fig, ax = plt.subplots()
        ax.contourf(X, Y, dist < 0, levels=100)
        plt.show()
