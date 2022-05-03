import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

X = np.random.randn(1000, 3)
X[:, 1] *= 4.0
X[:, 2] *= 16.0

pca = PCA(n_components=2)
pca.fit(X)

X_projected = pca.transform(X)
X_reconstructed = pca.inverse_transform(X_projected)

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='r')
ax.scatter(X_reconstructed[:, 0], X_reconstructed[:, 1], X_reconstructed[:, 2], c='b')
#ax.set_aspect('equal')
plt.show()
