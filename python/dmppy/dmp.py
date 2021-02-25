import numpy as np
#from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import math
from utils import midpoints
from utils import differentiate_seq

class MassSpringDamperSystem(object):
    def __init__(self, alpha=25.0, beta=5.0, center=0.0):
        self.alpha = alpha
        self.beta = beta
        self.center = center

    def acceleration(self, pos, vel):
        return self.alpha * (self.beta * (self.center - pos) - vel)

class CanonicalSystem(object):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def phase(self, t):
        return np.exp(-self.alpha * t)

    def basis_centers(self, n_basis):
        """
        return center of basis function (in phase space, not in time space) 
        """
        t_almost_converge = 5.0/self.alpha
        print(t_almost_converge)

        t_seq_aug = np.linspace(0, t_almost_converge, n_basis + 1)
        t_seq = midpoints(t_seq_aug)
        phase_seq = np.array([self.phase(t) for t in t_seq])
        return phase_seq

class DynamicalMovementPrimitive(object):

    def __init__(self, n_feature=100):
        self.system = MassSpringDamperSystem()
        self.canonical = CanonicalSystem()
        self.basis_centers = self.canonical.basis_centers(n_feature)

        tmp = np.diff(self.basis_centers)
        self.rbf_length = np.hstack([tmp, tmp[-1]])

        self.n_feature = n_feature

        self.weights = None
        self.pos_start = None
        self.pos_goal = None

    def _rbf_vals(self, phase):
        rbf = lambda diff: np.exp(- 1.0/(1.0 * self.rbf_length**2) * diff**2)
        diffs = phase - self.basis_centers
        return rbf(diffs)

    def _forcing_term(self, phase):
        rbf_vals = self._rbf_vals(phase)
        left_term = rbf_vals.dot(self.weights)/np.sum(self.weights)
        return left_term * phase * (self.pos_goal - self.pos_start)

    def fit(self, pos_seq, t_end):
        """
        pos_seq: trajectory numpy.array 1dim with equality separated time steps
        n_feature : int
        """

        n_step_ = len(pos_seq)
        dt = t_end / (n_step_ - 1)
        t_seq_pos = np.linspace(0, t_end, n_step_)
        t_seq_vel = midpoints(t_seq)
        t_seq_acc = midpoints(t_seq_vel)

        vel_seq = differentiate_seq(pos_seq, dt)
        acc_seq = differentiate_seq(vel_seq, dt)

        # adjust size of pos_seq and vel_seq to the same as acc_seq
        pos_seq = midpoints(midpoints(pos_seq))
        vel_seq = midpoints(vel_seq)

        system_acc_seq = [self.system.acceleration(pos, vel) for pos, vel in zip(pos_seq, vel_seq)]

        force_desired = acc_seq - system_acc_seq

        # create least square problem : minimize_w ||Aw = b||^2
        pos_start = pos_seq[0]
        pos_goal = pos_seq[-1]

        n_step = len(force_desired)
        phase_seq = [self.canonical.phase(t) for t in t_seq_acc]

        A = np.zeros((n_step, self.n_feature))
        for i in range(n_step):
            phase = phase_seq[i]
            xgy = phase * (pos_goal - pos_start) # term of x * (g - y0) in the paper
            rbf_vals = self._rbf_vals(phase)
            A[i, :] = rbf_vals/np.sum(rbf_vals) * xgy
        b = force_desired

        w = np.linalg.inv(A.T.dot(A) + 10.0 * np.eye(self.n_feature)).dot(A.T).dot(b)
        self.weights = w
        self.pos_start = pos_start
        self.pos_goal = pos_goal

        print(A.dot(w))
        print(b)



if __name__=='__main__':
    t_end = 3.14
    t_seq = np.linspace(0, t_end, 100)
    pos_seq = np.cos(t_seq)
    dmp = DynamicalMovementPrimitive(n_feature=200)
    dmp.fit(pos_seq, t_end)

    phase_lin = [dmp.canonical.phase(t) for t in t_seq]
    force_seq = [dmp._forcing_term(x) for x in phase_lin]
    plt.plot(t_seq, force_seq)
    plt.show()




