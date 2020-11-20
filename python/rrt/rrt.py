import numpy as np
import matplotlib.pyplot as plt 

np.random.seed(0)

class ConfigurationSpace(object):
    def __init__(self, b_min, b_max):
        self.b_min = b_min
        self.b_max = b_max
        self.n_dof = len(b_min)

    def sample(self):
        w = self.b_max - self.b_min
        return np.random.rand(self.n_dof) * w + self.b_min

class RapidlyExploringRandomTree(object): 
    def __init__(self, cspace, q_start, q_goal, sdf, 
            forward_kinematics= lambda Q: Q,
            N_maxiter=10000):
        self.cspace = cspace
        self.eps = 0.1
        self.n_resolution = 10
        self.N_maxiter = N_maxiter
        self.q_goal = np.array(q_goal)
        self.sdf = sdf
        self.fk = forward_kinematics

        # reserving memory is the key 
        self.Q_sample = np.zeros((N_maxiter, cspace.n_dof))
        self.idxes_parents = np.zeros(N_maxiter, dtype='int64')

        # set initial sample
        self.n_sample = 1
        self.Q_sample[0] = q_start
        self.idxes_parents[0] = 0 # self reference

    @property
    def q_start(self):
        return self.X_sample[0]

    def extend(self):
        def unit_vec(vec):
            return vec/np.linalg.norm(vec)

        q_rand = self.cspace.sample()
        q_rand_copied = np.repeat(q_rand.reshape(1, -1), self.n_sample, axis=0)

        sqdists = np.sum((self.Q_sample[:self.n_sample] - q_rand_copied)**2, axis=1)
        idx_nearest = np.argmin(sqdists)
        q_nearest = self.Q_sample[idx_nearest]
        if np.linalg.norm(q_rand - q_nearest) > self.eps:
            q_new = q_nearest + unit_vec(q_rand - q_nearest) * self.eps
        else:
            q_new = q_rand

        # update tree
        if self.sdf(self.fk(q_new)) > 0:
            self.Q_sample[self.n_sample] = q_new
            self.idxes_parents[self.n_sample] = idx_nearest
            self.n_sample += 1

    def show(self):
        fig, ax = plt.subplots()
        n = self.n_sample
        ax.scatter(self.Q_sample[:n, 0], self.Q_sample [:n, 1], c="black")
        for q, parent_idx in zip(self.Q_sample[:n], self.idxes_parents[:n]):
            q_parent = self.Q_sample[parent_idx]
            ax.plot([q[0], q_parent[0]], [q[1], q_parent[1]], color="red")


if __name__=='__main__':
    b_min = np.zeros(2)
    b_max = np.ones(2)
    cspace = ConfigurationSpace(b_min, b_max)
    sdf = lambda q: np.linalg.norm(q - np.array([0.5, 0.5])) - 0.3
    rrt = RapidlyExploringRandomTree(cspace, [0.1, 0.1], [0.9, 0.9], sdf)
    import time
    ts = time.time()
    for i in range(2000):
        rrt.extend()
    print(time.time() - ts)
    rrt.show()
    plt.show()
