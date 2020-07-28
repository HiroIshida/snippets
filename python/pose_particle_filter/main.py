import scipy.stats
import numpy as np

class ParticleFilter:
    def __init__(self, N, M=3):
        self.N = N
        self.M = M

    def initialize(self, X, W=None):
        N = X.shape[0]
        self.X = X
        W = np.ones(N)/self.N
        self.W = W

    def default_likelihood_function(self, X, z, R):
        Rinv = np.linalg.inv(R)
        Rdet = np.linalg.det(R)
        d = X - z.reshape(1, self.M).repeat(self.N, axis=0)
        tmp = np.einsum('ij,jk', d, Rinv) 
        dists = np.sum(tmp.T * d.T, axis=0)
        return np.exp(-0.5 * dists)

    def update(self, z, R):
        weights_likeli = self.default_likelihood_function(self.X, z, R)
        tmp = self.W * weights_likeli
        self.W = tmp/sum(tmp) 

def mv_normal(X, Z, R):
    Rinv = np.linalg.inv(R)
    Rdet = np.linalg.det(R)
    d = X - Z
    tmp = np.einsum('ij,jk', d, Rinv) 
    dists = np.sum(tmp.T * d.T, axis=0)
    #return weights = np.exp(-0.5 * dists)

if __name__=='__main__':
    import numpy as np
    N = 10000
    X = np.random.randn(N, 3) 
    pf = ParticleFilter(N)
    pf.initialize(X)
    pf.update(np.array([0.5, 0.5, 0.5]), np.eye(3))

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    print(sum(pf.W))

    myscat = lambda X, color: ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color)
    myscat(X, 'red')
    plt.show()
