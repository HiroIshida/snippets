from sklearn.cluster import SpectralClustering
import numpy as np
import matplotlib.pyplot as plt

# create block matrix
n_sample = 40
n_sample_half = 20
X = np.zeros((n_sample, n_sample), dtype=int)
X[:n_sample_half, :n_sample_half] = 1
X[n_sample_half:, n_sample_half:] = 1

for k in range(5):
    for i in range(n_sample - k):
        X[i, i + k] = 1
        X[i + k, i] = 1

add = np.random.randint(0, 2, (n_sample, n_sample))
X = (X + add) // 2

for k in range(n_sample):
    X[k, k] = 0

C = SpectralClustering(n_clusters=2, assign_labels="discretize", random_state=0)
clustering = C.fit(X)
print(clustering.labels_)

plt.imshow(X)
plt.show()
