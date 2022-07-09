import numpy as np
from scipy import sparse
import networkx as nx

def get_python_modsoft_object(graph, learning_rate=1., init_part=None,
                              n_communities=None, bias=0., resolution=1.):

    if type(graph) == sparse.csr_matrix:
        adj_matrix = graph
    elif type(graph) == np.ndarray:
        adj_matrix = sparse.csr_matrix(graph)
    elif type(graph) == nx.classes.graph.Graph:
        adj_matrix = nx.adj_matrix(graph)
    else:
        raise TypeError("The argument should be a Numpy Array or a Compressed Sparse Row Matrix.")

    if init_part is None:
        init_part = np.arange(adj_matrix.shape[0], dtype=np.int)

    if n_communities is None:
        n_communities = adj_matrix.shape[0]

    return PythonModSoft(adj_matrix.shape[0],
                         adj_matrix.indices, adj_matrix.indptr, np.array(adj_matrix.data, dtype=np.float),
                         n_communities, learning_rate, bias, init_part, resolution)

def project_top_K(in_map, K):
    cum_sum = 0
    v_max = 0.
    lamb = 0
    out_map = dict()
    i = 0
    while len(in_map) > 0 and i < K:
        first = True
        for j, v in in_map.items():
            if first or v > v_max:
                v_max = v
                j_max = j
                first = 0
        in_map.pop(j_max, None)
        cum_sum += v_max
        new_lamb = (1. / (i + 1.)) * (1. - cum_sum)
        if v_max + new_lamb <= 0.:
            break
        else:
            out_map[j_max] = v_max
            lamb = new_lamb
            i += 1

    for i, v in out_map.items():
        out_map[i] = max(v + lamb, 0)

    return out_map


class PythonModSoft(object):

    def __init__(self, n_nodes, indices, indptr, data, n_com, learning_rate, bias, init_part, resolution):
        self.n_com = n_com
        self.learning_rate = learning_rate
        self.bias = bias

        self.resolution = resolution

        self.n_nodes = n_nodes
        self.graph_edges = dict()
        self.degree = dict()
        self.w = 0
        for node in range(n_nodes):
            self.graph_edges[node] = dict()
            self.degree[node] = 0
            for i in range(indptr[node], indptr[node + 1]):
                self.graph_edges[node][indices[i]] = data[i]
                if indices[i] == node:
                    self.degree[node] += 2. * data[i]
                    self.w += 2. * data[i]
                else:
                    self.degree[node] += data[i]
                    self.w += data[i]
        self.w_inv = 1. / self.w

        self.p = dict()
        self.avg_p = [0 for i in range(self.n_nodes)]
        for node in range(n_nodes):
            self.p[node] = dict()
            self.p[node][init_part[node]] = 1.
            self.avg_p[init_part[node]] += self.w_inv * self.degree[node]

    def modularity(self):
        Q = 0.
        for com in range(self.n_nodes):
            Q -= self.avg_p[com] * self.avg_p[com]
        for node in range(self.n_nodes):
            for neighbor, weight in self.graph_edges[node].items():
                if neighbor == node:
                    weight *= 2.
                for com, p_com in self.p[node].items():
                    if com in self.p[neighbor]:
                        Q += self.resolution * self.w_inv * weight * p_com * self.p[neighbor][com]
        return Q

    def one_step(self):

        for node in range(self.n_nodes):
            new_p = dict()
            for com, p_com in self.p[node].items():
                new_p[com] = self.bias * p_com
            for neighbor, weight in self.graph_edges[node].items():
                for com, p_com in self.p[neighbor].items():
                    if com not in new_p:
                        new_p[com] = 0
                    new_p[com] += self.resolution * self.learning_rate * weight * p_com
            for com, new_p_com in new_p.items():
                new_p[com] = new_p_com - self.learning_rate * self.degree[node] * self.avg_p[com]
            new_p = project_top_K(new_p, self.n_com)
            for com, p_com in self.p[node].items():
                self.avg_p[com] -= self.w_inv * self.degree[node] * p_com
            self.p[node] = dict()
            for com, new_p_com in new_p.items():
                self.avg_p[com] += self.w_inv * self.degree[node] * new_p_com
                self.p[node][com] = new_p_com

    def get_membership(self):
        return self.p
