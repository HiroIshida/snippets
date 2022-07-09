from community import community_louvain
from modsoft import get_python_modsoft_object

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# create block matrix
n_sample = 40
n_sample_half = 20
n_margin = 3
X = np.zeros((n_sample, n_sample), dtype=int)
X[:n_sample_half + n_margin, :n_sample_half + n_margin] = 1
X[n_sample_half - n_margin:, n_sample_half - n_margin:] = 1

add = np.random.randint(0, 2, (n_sample, n_sample))
X = (X + add) // 2

for k in range(n_sample):
    X[k, k] = 0

G = nx.from_numpy_matrix(X)
partition = community_louvain.best_partition(G)
ms = get_python_modsoft_object(G, init_part=partition, learning_rate=0.5)

for i in range(100):
    ms.one_step()
    score = ms.modularity()
    print(score)

membership = ms.get_membership()
print(membership)
