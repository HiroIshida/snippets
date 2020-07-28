import scipy.stats
import numpy as np
import math

def marging_resampling(ptcls, w, mp = 0.9, 
        unit_circle_idxes = [2],
        unit_circle_bminmaxes = [[0, 2*3.1415]]):
    N = len(w)
    idxes1 = low_variance_sampler(w)
    idxes2_ = low_variance_sampler(w)
    idxes3_ = low_variance_sampler(w)
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

def unit_circle_merge(s_lst, a_lst, bmin, bmax):

    def unit_circle_encode(s):
        tmp = (s - bmin) / (bmax - bmin)
        phase = tmp * 2 * math.pi
        x = np.vectorize(math.cos)(phase)
        y = np.vectorize(math.sin)(phase)
        return np.vstack((x, y))

    def stick_to_unit_circle(pos):
        r = np.vectorize(math.sqrt)(np.sum(pos**2, 0))
        return pos * (1/r)

    def unit_circle_decode(pos):
        phase  = np.arctan2(pos[1], pos[0]) 
        idx = np.where(phase < 0)[0]
        phase[idx] =  phase[idx] + 2 * math.pi
        s = (phase / (2 * math.pi)) * (bmax - bmin) + bmin
        return s

    pos0 = unit_circle_encode(s_lst[0])
    pos1 = unit_circle_encode(s_lst[1])
    pos2 = unit_circle_encode(s_lst[2])
    pos_tmp = a_lst[0] * pos0 + a_lst[1] * pos1 + a_lst[2] * pos2
    pos = stick_to_unit_circle(pos_tmp)
    s = unit_circle_decode(pos)
    return s

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
            self.X = marging_resampling(self.X, self.W)



if __name__=='__main__':
    import numpy as np
    N = 1000
    X = np.random.randn(N, 3) 
    pf = ParticleFilter(N)
    pf.initialize(X)
    pf.update(np.array([0.0, 0.0, 0.6]), np.eye(3)*0.2)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    print(sum(pf.W))

    myscat = lambda X, color: ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color)
    myscat(pf.X, 'red')
    plt.show()
