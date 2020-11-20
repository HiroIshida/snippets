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
    def __init__(self, cspace, x_start, x_goal, N_maxiter=10000):
        self.cspace = cspace
        self.eps = 0.1
        self.N_maxiter = N_maxiter
        self.x_goal = np.array(x_goal)

        # reserving memory is the key 
        self.X_sample = np.zeros((N_maxiter, cspace.n_dof))
        self.idxes_parents = np.zeros(N_maxiter, dtype='int64')

        # set initial sample
        self.n_sample = 1
        self.X_sample[0] = x_start
        self.idxes_parents[0] = 0 # self reference

    @property
    def x_start(self):
        return self.X_sample[0]

    def extend(self, debug=False):
        def unit_vec(vec):
            return vec/np.linalg.norm(vec)

        x_rand = self.cspace.sample()
        x_rand_copied = np.repeat(x_rand.reshape(1, -1), self.n_sample, axis=0)

        sqdists = np.sum((self.X_sample[:self.n_sample] - x_rand_copied)**2, axis=1)
        idx_nearest = np.argmin(sqdists)
        x_nearest = self.X_sample[idx_nearest]
        if np.linalg.norm(x_rand - x_nearest) > self.eps:
            x_new = x_nearest + unit_vec(x_rand - x_nearest) * self.eps
        else:
            x_new = x_rand

        # update tree
        self.X_sample[self.n_sample] = x_new
        self.idxes_parents[self.n_sample] = idx_nearest
        self.n_sample += 1

    def show(self):
        fig, ax = plt.subplots()
        n = self.n_sample
        ax.scatter(self.X_sample[:n, 0], self.X_sample [:n, 1], c="black")
        for x, parent_idx in zip(self.X_sample[:n], self.idxes_parents[:n]):
            x_parent = self.X_sample[parent_idx]
            ax.plot([x[0], x_parent[0]], [x[1], x_parent[1]], color="red")


if __name__=='__main__':
    b_min = np.zeros(2)
    b_max = np.ones(2)
    cspace = ConfigurationSpace(b_min, b_max)
    rrt = RapidlyExploringRandomTree(cspace, [0.1, 0.1], [0.9, 0.9])
    import time
    ts = time.time()
    for i in range(2000):
        rrt.extend(debug=False)
    print(time.time() - ts)
    rrt.show()
    plt.show()
