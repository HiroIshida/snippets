import GPy
import numpy as np
import matplotlib.pyplot as plt

# set variance larger
kernel = GPy.kern.RBF(1, lengthscale=2.0, variance=10.0)

X = np.random.randn(30)
Y = np.vstack([np.cos(X), np.sin(X)]).T
X = np.expand_dims(X, axis=0).T

model = GPy.models.GPRegression(X, Y, kernel=kernel)
model.plot()
plt.show()
