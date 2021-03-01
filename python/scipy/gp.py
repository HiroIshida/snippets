import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T
Y = np.array([-1, -1, -1, 1, 1, -1.])
lscale = 1.0
noise = 0.1
kernel = RBF(1.0)
gp = GaussianProcessRegressor(kernel=kernel, alpha=noise, optimizer=None)
gp.fit(X, Y)

x_test =np.atleast_2d(np.linspace(0, 10, 100)).T
y_test, sigma = gp.predict(x_test, return_std=True)

plt.plot(x_test, y_test)
plt.fill_between(x_test.flatten(), y_test - 2 * sigma, y_test + 2 * sigma, alpha=0.2)
plt.scatter(X.flatten(), Y.flatten())
plt.show()
