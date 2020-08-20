import numpy as np
import matplotlib.pyplot as plt 
from skimage import measure

def sdf_sphere(X, c, r):
    n_points = X.shape[0]
    C = np.repeat(c.reshape(1, 2), n_points, axis=0)
    dists = np.sqrt(np.sum((X - C)**2, 1))
    return dists - r

def sdf_combine(X):
    f1 = sdf_sphere(X, np.array([0, -0.3]), 0.6)
    f2 = sdf_sphere(X, np.array([0, 0.4]), 0.4)
    logicals = f1 > f2
    return f2 * logicals + f1 * (~logicals)

class SampleTestData:
    def __init__(self, n_grid=100, n_interval=5):
        ns = np.array([n_grid, n_grid])
        b_min = np.array([-1.0, -1.0])
        b_max = np.array([1.0, 1.0])
        xlin, ylin = [np.linspace(b_min[i], b_max[i], ns[i]) for i in range(2)]
        X, Y = np.meshgrid(xlin, ylin)

        pts = np.array(zip(X.flatten(), Y.flatten()))
        fs_ = sdf_combine(pts)
        fs = fs_.reshape(ns)

        c__ = measure.find_contours(fs.T, 0.0)[0] # only one closed curve now
        def rescale_contour(pts, b_min, b_max, n):
            n_points, n_dim = pts.shape
            width = b_max - b_min
            b_min_tile = np.tile(b_min, (n_points, 1))
            width_tile = np.tile(width, (n_points, 1))
            pts_rescaled = b_min_tile + width_tile * pts / (n - 1)
            return pts_rescaled

        c_ = c__[::n_interval]
        c = rescale_contour(c_, b_min, b_max, ns[0])

        self.b_min = b_min
        self.b_max = b_max
        self.X = X
        self.Y = Y
        self.fs = fs
        self.c = c
        self.ns = ns

    def show(self, data=None):
        if data is None:
            data = self.fs
        fig, ax = plt.subplots()
        cplt = ax.contourf(self.X, self.Y, data)
        cbar = fig.colorbar(cplt)
        ax.scatter(self.c[:, 0], self.c[:, 1])
        plt.show()

if __name__=='__main__':
    std = SampleTestData()
    std.show()
