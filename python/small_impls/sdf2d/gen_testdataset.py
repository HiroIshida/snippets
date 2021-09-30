import numpy as np
import matplotlib.pyplot as plt 
from skimage import measure
import json

def sdf_sphere(X, c, r):
    n_points = X.shape[0]
    C = np.repeat(c.reshape(1, 2), n_points, axis=0)
    dists = np.sqrt(np.sum((X - C)**2, 1))
    return dists - r

def sdf_snowball(X):
    f1 = sdf_sphere(X, np.array([0, -0.3]), 0.6)
    f2 = sdf_sphere(X, np.array([0, 0.4]), 0.4)
    logicals = f1 > f2
    return f2 * logicals + f1 * (~logicals)

def sdf_multiple(X):
    f1 = sdf_sphere(X, np.array([0.5, 0.5]), 0.3)
    f2 = sdf_sphere(X, np.array([-0.5, -0.5]), 0.3)
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
        fs_ = sdf_multiple(pts)
        fs = fs_.reshape(ns)

        c_list = measure.find_contours(fs.T, 0.0)
        def rescale_contour(pts, b_min, b_max, n):
            n_points, n_dim = pts.shape
            width = b_max - b_min
            b_min_tile = np.tile(b_min, (n_points, 1))
            width_tile = np.tile(width, (n_points, 1))
            pts_rescaled = b_min_tile + width_tile * pts / (n - 1)
            return pts_rescaled

        process = lambda c: rescale_contour(c[::n_interval], b_min, b_max, ns[0])
        V_list = map(process, c_list) # V is list of vertex, V_list is list of V

        def make_edges(V):
            n_vert = len(V)
            E = np.array(zip(range(n_vert), np.array(range(n_vert)) + 1))
            E[n_vert-1][1] = 0 
            return E
        E_list = map(make_edges, V_list)

        V = V_list[0]; E = E_list[0]
        for i in range(len(E_list)-1):
            E_append = E_list[i+1] + len(V)
            E = np.vstack((E, E_append))
            V_append = V_list[i+1]
            V = np.vstack((V, V_append))

        self.b_min = b_min
        self.b_max = b_max
        self.X = X
        self.Y = Y
        self.fs = fs
        self.V = V
        self.E = E
        self.ns = ns

    def show(self, data=None):
        if data is None:
            data = self.fs
        fig, ax = plt.subplots()
        cplt = ax.contourf(self.X, self.Y, data)
        cbar = fig.colorbar(cplt)
        ax.scatter(self.V[:, 0], self.V[:, 1])
        plt.show()

    def save(self):
        data = {}
        data["b_min"] = list(self.b_min)
        data["b_max"] = list(self.b_max)
        data["N"] = list(self.ns)
        data["V"] = [list(vert) for vert in self.V] 
        data["E"] = [list(edge) for edge in self.E] 
        with open("test.json", 'w') as f:
            json.dump(data, f)

if __name__=='__main__':
    std = SampleTestData()
    std.save()
    std.show()
