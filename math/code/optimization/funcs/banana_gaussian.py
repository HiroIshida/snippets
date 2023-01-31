import matplotlib.pyplot as plt 
import numpy as np

def banana_gaussian(pts):
    x_lst = pts[:, 0]
    y_lst = pts[:, 1]

    r_lst = np.sqrt(x_lst**2 + y_lst**2) - 1
    theta_lst = np.arctan2(y_lst, x_lst)

    pts_ = np.vstack((theta_lst, r_lst)).T

    Sigma = np.diag((2**2, 0.3**2))
    tmp= pts_.dot(np.linalg.inv(Sigma))
    sqdist = np.sum(tmp * pts_, axis=1)
    return np.exp(-0.5 * sqdist)

N_grid = 100
xlin = np.linspace(-3.0, 3.0, N_grid)
ylin = np.linspace(-3.0, 3.0, N_grid)
X_mesh, Y_mesh = np.meshgrid(xlin, ylin)
pts = np.array(list(zip(X_mesh.flatten(), Y_mesh.flatten())))
Z = banana_gaussian(pts).reshape((100, 100))

fig, ax = plt.subplots()
ax.contourf(X_mesh, Y_mesh, Z)
plt.show()


