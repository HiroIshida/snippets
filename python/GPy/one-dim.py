import GPy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# set variance larger
kernel = GPy.kern.RBF(1, lengthscale=2.0, variance=10.0)

X = np.array([[-5, -3, 0, 3, 5.]]).T
Y = np.array([[-1, -1, 1.1, 1, -1.]]).T
model = GPy.models.GPRegression(X, Y, kernel=kernel)
model.plot()
plt.show()

## prediction
x_pred = np.linspace(-10, 10, 100)
x_pred = x_pred[:, None]
y_qua_pred = model.predict_quantiles(x_pred, quantiles=(2.5, 50, 97.5))[0]
