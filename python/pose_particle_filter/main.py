import scipy.stats
import numpy as np

def low_variance_sampler(ws):
    # ws is numpya array
    N = len(ws)
    idxes = np.zeros(N, dtype=int)
    w_sum = sum(ws)
    r = np.random.random()*(1.0/N)
    c = ws[0]/w_sum
    k = 0
    for n in range(N):
        U = r + n*(1.0/N)
        while U > c:
            k+=1
            c = c+ws[k]/w_sum
        idxes[n] = k
    return idxes

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

    def update(self, z, R, resampling=True):
        weights_likeli = self.default_likelihood_function(self.X, z, R)
        tmp = self.W * weights_likeli
        self.W = tmp/sum(tmp) 
        if resampling:
            idxes = low_variance_sampler(self.W)
            print(idxes)
            self.X = self.X[idxes, :]
            self.W = np.ones(self.N)/self.N



if __name__=='__main__':
    import numpy as np
    N = 1000
    X = np.random.randn(N, 3) 
    pf = ParticleFilter(N)
    pf.initialize(X)
    pf.update(np.array([2.0, 0.0, 0.0]), np.eye(3)*0.2)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    print(sum(pf.W))

    myscat = lambda X, color: ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color)
    myscat(pf.X, 'red')
    plt.show()
