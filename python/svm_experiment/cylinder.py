from sklearn import svm
from sklearn.metrics import pairwise
import matplotlib.pyplot as plt 
import numpy as np
import math

def circle_metric(X_sub, Y_sub): # circle subspace
    X_vec = np.vstack([np.cos(X_sub), np.sin(X_sub)]).T
    Y_vec = np.vstack([np.cos(Y_sub), np.sin(Y_sub)]).T
    inpros = np.sum(X_vec * Y_vec, axis=1) # cos(theta)
    return np.arccos(inpros)

def cylinder_metric(X, Y): 
    dist_circle = circle_metric(X[:,  0], Y[:, 0])
    dist_euclidan = np.abs(Y[:, 1] - X[:, 1])
    return np.sqrt(dist_circle ** 2 + dist_euclidan ** 2)

def plot_cylinder_coordinate(center):
    N = 100
    lin_circle = np.linspace(-math.pi, math.pi, N)
    lin_euclidean = np.linspace(-2.0, 2.0, N)
    mesh_grid = np.meshgrid(lin_circle, lin_euclidean)
    pts = np.array(zip(mesh_grid[0].flatten(), mesh_grid[1].flatten()))
    
    center_copied = np.repeat(np.array([center]), pts.shape[0], axis=0)
    dists = cylinder_metric(pts, center_copied)
    dist_grid = dists.reshape(N, N)

    fig, ax = plt.subplots()
    cs = ax.contourf(lin_circle, lin_euclidean, dist_grid, cmap = 'jet', zorder=1)
    plt.show()

def generate_cylinder_kernel(gammas = None):
    # as for closure in py2x see:
    # https://stackoverflow.com/questions/3190706/nonlocal-keyword-in-python-2-x
    d = {'gammas': gammas}
    def kern(X_, Y_ = None):
        gammas = d['gammas']
        n_x, m_x = X_.shape
        assert m_x == 2

        if gammas is None:
            gammas = np.ones(m_x)
        if Y_ is None:
            Y_ = X_
        n_y = Y_.shape[0]

        mat = np.zeros((n_x, n_y))
        for j in range(n_y):
            Y_repeated = np.repeat(np.array([Y_[j]]), n_x, axis=0)
            dists = cylinder_metric(X_, Y_repeated)
            mat[:, j] = np.exp(- 0.01 * dists**2)
        return mat
    return kern

#plot_cylinder_coordinate([0.5, 0.5])

use_svm = True
if use_svm:
    dataset1 = [
            [[0, 0], [0, 0], [1, 1]],
            [0, 0, 1]
            ]
    dataset2 = [
            [[0, 0], [1., 1]],
            [0, 1]
            ]
    kern = generate_cylinder_kernel()
    clf = svm.SVC(kernel = kern)
    myfit = lambda dataset : clf.fit(dataset[0], dataset[1])
    myfit(dataset2)

    N = 100
    lin_circle = np.linspace(-math.pi, math.pi, N)
    lin_euclidean = np.linspace(-4.0, 4.0, N)
    mesh_grid = np.meshgrid(lin_circle, lin_euclidean)
    pts = np.array(zip(mesh_grid[0].flatten(), mesh_grid[1].flatten()))
    preds_ = clf.decision_function(pts)
    preds = preds_.reshape(N, N)

    fig, ax = plt.subplots()
    cs = ax.contourf(lin_circle, lin_euclidean, preds, cmap = 'jet')
    plt.show()

