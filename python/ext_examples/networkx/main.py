from community import community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
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

G = nx.from_numpy_matrix(X)
#pos = nx.spring_layout(G)

# compute the best partition
partition = community_louvain.best_partition(G)
#from IPython import embed; embed()

#
## draw the graph
pos = nx.spring_layout(G)
cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
                       cmap=cmap, node_color=list(partition.values()))
nx.draw_networkx_edges(G, pos, alpha=0.5)
plt.show()
