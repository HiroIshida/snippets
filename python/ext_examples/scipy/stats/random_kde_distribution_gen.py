import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

np.random.seed(2)
width = np.array([0.6, 1.0])
margin = 0.15
points = np.random.rand(10, 2) * (width - margin * 2) + margin
kde = gaussian_kde(points.T)

x = np.linspace(0, 0.6, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
Z = kde.evaluate(np.vstack([X.ravel(), Y.ravel()]))
Z = Z.reshape(X.shape)

points_resampled = kde.resample(100).T
fig, ax = plt.subplots()
ax.contour(X, Y, Z)
plt.show()
