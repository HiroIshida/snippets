import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

X = np.random.randn(100, 2)
X[:, 0] *= 4.0

pca = PCA(n_components=1)
pca.fit(X)

X_projected = pca.transform(X)
X_reconstructed = pca.inverse_transform(X_projected)

fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c='r')
ax.scatter(X_reconstructed[:, 0], X_reconstructed[:, 1], c='b')
ax.set_aspect('equal')
plt.show()
