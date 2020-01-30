import GPy
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0., 0], [1, 2], [-2, 1]])
Y = np.array([[1, 0 ,0]]).T
model = GPy.models.GPRegression(X, Y)
model.plot()
model.predict(np.array([[0, 0.01]]))

