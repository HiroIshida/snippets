import GPy
import numpy as np
import matplotlib.pyplot as plt
import copy

X_posi = np.array([
    [0., 1.0], 
    [0., -1.0], 
    [1., 1.2],
    [1.0, -1.4]])

X_nega = np.array([
    [0., 1.2], 
    [0., -1.2], 
    [1., 1.4],
    [1.0, -1.6]])

X = np.vstack((X_posi, X_nega))
Y = np.array([[1, 1, 1, 1., 0., 0, 0, 0]]).T

kernel = GPy.kern.Matern52(2, variance=0.01, lengthscale=0.8)
model = GPy.models.GPClassification(X, Y, kernel=kernel)
#model.plot()

def show2d(func, bmin, bmax, N = 20, fax = None, levels = None):
    # fax is (fix, ax)
    # func: np.array([x1, x2]) list -> scalar
    # func cbar specifies the same height curve
    if fax is None:
        fig, ax = plt.subplots() 
    else:
        fig = fax[0]
        ax = fax[1]

    mat_contour_ = np.zeros((N, N))
    x1_lin, x2_lin = [np.linspace(bmin[i], bmax[i], N) for i in range(bmin.size)]
    for i in range(N):
        for j in range(N):
            x = np.array([[x1_lin[i], x2_lin[j]]])
            val = func(x)
            mat_contour_[i, j] = val
    mat_contour = mat_contour_.T
    X, Y = np.meshgrid(x1_lin, x2_lin)

    cs = ax.contour(X, Y, mat_contour, levels = levels, cmap = 'jet')
    zc = cs.collections[0]
    plt.setp(zc, linewidth=4)
    ax.clabel(cs, fontsize=10)
    cf = ax.contourf(X, Y, mat_contour, cmap = 'gray_r')
    fig.colorbar(cf)

b_min, b_max = model.get_boundary(margin = 1.0)
import pdb; pdb.set_trace()
model.predict(np.array([[0.2, 0]]))

"""
def f(x):
    print x
    mu, cov = model.predict(np.array(x))
    print cov
    return cov.item()

show2d(f, b_min, b_max, levels = [])
#model.predict(np.array([[0, 0.01]]))

"""
