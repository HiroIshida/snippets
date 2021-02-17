from sklearn import svm
from sklearn.metrics import pairwise
import matplotlib.pyplot as plt 
import numpy as np
import math
from math import *

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
            mat[:, j] = np.exp(- 10 * dists**2)
        return mat
    return kern

#plot_cylinder_coordinate([0.5, 0.5])

use_svm = True
if use_svm:
    slide = 2.0 
    radius = 1.0
    scale = 1.0
    n_positive = 20
    X_positive = np.array([[radius * cos(2 * pi * i/n_positive), radius * sin(2 * pi * i/n_positive)] for i in range(n_positive)])
    X_positive[:, 0] *= scale
    X_positive[:, 0] += slide
    Y = [1] * n_positive

    n_negative = 20
    X_negative = []
    while(True):
        x_cand = np.random.rand(2) * 2 - np.ones(2)
        if np.sqrt(x_cand[0] ** 2 + x_cand[1] ** 2) < radius * 0.8:
            X_negative.append(x_cand)
        if len(X_negative) == n_negative:
            break
    X_negative = np.array(X_negative)
    X_negative[:, 0] *= scale
    X_negative[:, 0] += slide
    X = np.vstack((X_positive, X_negative))
    Y.extend([0] * n_negative)
    dataset = [X, Y]
    use_cylinde = True
    if use_cylinde:
        kern = generate_cylinder_kernel()
        clf = svm.SVC(kernel = kern, C=100)
    else:
        clf = svm.SVC()
    myfit = lambda dataset : clf.fit(dataset[0], dataset[1])
    myfit(dataset)

    N = 100
    lin_circle = np.linspace(-math.pi, math.pi, N)
    lin_euclidean = np.linspace(-4.0, 4.0, N)
    mesh_grid = np.meshgrid(lin_circle, lin_euclidean)
    pts = np.array(zip(mesh_grid[0].flatten(), mesh_grid[1].flatten()))
    preds_ = clf.decision_function(pts)
    preds = preds_.reshape(N, N)

    fig, ax = plt.subplots()
    cs = ax.contourf(lin_circle, lin_euclidean, preds, cmap = 'jet')
    ax.scatter(X_negative[:, 0], X_negative[:, 1], c="red")
    ax.scatter(X_positive[:, 0], X_positive[:, 1], c="blue")
    plt.show()

