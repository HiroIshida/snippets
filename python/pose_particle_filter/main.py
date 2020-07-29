import scipy.stats
import numpy as np
import math
from low_variance_sampler import low_variance_sampler_cython

def marging_resampling(ptcls, w, mp = 0.9, 
        unit_circle_idxes = [2],
        unit_circle_bminmaxes = [[0, 2*3.1415]],
        cython=False
        ):
    N = len(w)

    sampler = low_variance_sampler_c if cython else low_variance_sampler
    idxes1 = sampler(w)
    idxes2_ = sampler(w)
    idxes3_ = sampler(w)
    perm2 = np.random.permutation(N)
    perm3 = np.random.permutation(N)
    idxes2 = idxes2_[perm2]
    idxes3 = idxes3_[perm3]
    a1 = mp
    a2 = (0.5 * ((1 - a1) + math.sqrt(-3 * a1**2 + 2 * a1 + 1)))
    a3 = (0.5 * ((1 - a1) - math.sqrt(-3 * a1**2 + 2 * a1 + 1)))
    ptcl_tmp1 = ptcls[idxes1, :]
    ptcl_tmp2 = ptcls[idxes2, :]
    ptcl_tmp3 = ptcls[idxes3, :]
    ptcls = ptcl_tmp1 * a1 + ptcl_tmp2 * a2 + ptcl_tmp3 * a3

    N_unit_circle_merge = len(unit_circle_idxes)
    for i in range(N_unit_circle_merge):
        axis = unit_circle_idxes[i]
        ptcl_partial_tmp1 = ptcl_tmp1[:, axis]
        ptcl_partial_tmp2 = ptcl_tmp2[:, axis]
        ptcl_partial_tmp3 = ptcl_tmp3[:, axis]

        idx_ucm = unit_circle_idxes[i]
        bmin_ucm = unit_circle_bminmaxes[i][0]
        bmax_ucm = unit_circle_bminmaxes[i][1]
        ptcls[:, axis] = unit_circle_merge(
                [ptcl_partial_tmp1, ptcl_partial_tmp2, ptcl_partial_tmp3],
                [a1, a2, a3], bmin_ucm, bmax_ucm)
    return ptcls

def low_variance_sampler_c(ws):
    r = np.random.random()
    idxes = low_variance_sampler_cython(ws, r)
    return idxes

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

def unit_circle_encode(s, bmin, bmax):
    tmp = (s - bmin) / (bmax - bmin)
    phase = tmp * 2 * math.pi
    x = np.vectorize(math.cos)(phase)
    y = np.vectorize(math.sin)(phase)
    return np.vstack((x, y))

def stick_to_unit_circle(pos):
    r = np.vectorize(math.sqrt)(np.sum(pos**2, 0))
    return pos * (1/r)

def unit_circle_decode(pos, bmin, bmax):
    phase  = np.arctan2(pos[1], pos[0]) 
    idx = np.where(phase < 0)[0]
    phase[idx] =  phase[idx] + 2 * math.pi
    s = (phase / (2 * math.pi)) * (bmax - bmin) + bmin
    return s

def regularize(angles, bmin, bmax):
    tmp = (angles - bmin)/(bmax - bmin)
    tmp2 = tmp - np.floor(tmp)
    return tmp2 * (bmax - bmin) + bmin

def unit_circle_merge(s_lst, a_lst, bmin, bmax):
    pos0 = unit_circle_encode(s_lst[0], bmin, bmax)
    pos1 = unit_circle_encode(s_lst[1], bmin, bmax)
    pos2 = unit_circle_encode(s_lst[2], bmin, bmax)
    pos_tmp = a_lst[0] * pos0 + a_lst[1] * pos1 + a_lst[2] * pos2
    pos = stick_to_unit_circle(pos_tmp)
    s = unit_circle_decode(pos, bmin, bmax)
    return s

class ParticleFilter:
    def __init__(self, N, M=3):
        self.N = N
        self.M = M

    def initialize(self, X, W=None):
        N = X.shape[0]
        self.X = X
        self.X[:, 2] = regularize(X[:, 2], 0, 2*3.1415)
        W = np.ones(N)/self.N
        self.W = W

    def default_likelihood_function(self, X, z, R):
        Rinv = np.linalg.inv(R)
        Rdet = np.linalg.det(R)
        d = X - z.reshape(1, self.M).repeat(self.N, axis=0)
        tmp = np.einsum('ij,jk', d, Rinv) 
        dists = np.sum(tmp.T * d.T, axis=0)
        return np.exp(-0.5 * dists)

    def update(self, z, R, resampling=True, ess_threshold=0.99999):
        weights_likeli = self.default_likelihood_function(self.X, z, R)
        tmp = self.W * weights_likeli
        self.W = tmp/sum(tmp) 

        ess = 1.0/(np.sum(self.W ** 2) * self.N) # effective sample size
        if resampling:
            if ess < ess_threshold:
                message = "effective sample size is {0} < {1}, thus the resampling took place".format(\
                        ess, ess_threshold)
                print(message)
                self.X = marging_resampling(self.X, self.W)
                self.W = np.ones(self.N)/self.N

    def get_current_est(self):
        x_mean = np.dot(self.X.T, self.W) 
        diff = self.X - x_mean.reshape(1, 3).repeat(self.N, axis=0)
        diff_weighted = diff * self.W.reshape(self.N, 1).repeat(3, axis=1)
        cov = np.einsum('ki,kj', diff, diff_weighted)
        return x_mean, cov

if __name__=='__main__':
    import numpy as np
    N = 10000
    X = np.random.randn(N, 3) * 3
    pf = ParticleFilter(N)
    pf.initialize(X)
    import time
    ts = time.time()
    for i in range(1):
        pf.update(np.array([0.0, 0.0, 0.6]), np.eye(3)*0.9)
    print(time.time() - ts)

    print(pf.get_current_est())


    vistest = False
    if vistest:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        print(sum(pf.W))

        myscat = lambda X, color: ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color)
        myscat(pf.X, 'red')
        plt.show()
