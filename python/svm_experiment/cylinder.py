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

plot_cylinder_coordinate([1.3, 0])

