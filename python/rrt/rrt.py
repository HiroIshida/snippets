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
    def __init__(self, cspace, x_start, x_goal):
        self.cspace = cspace
        self.x_goal = np.array(x_goal)
        self.X_sample = [np.array(x_start)]
        self.idxes_parents = [0] # 0 indicates to self-reference
        self.eps = 0.1

    @property
    def x_start(self):
        return self.X_sample[0]

    def extend(self):
        def unit_vec(vec):
            return vec/np.linalg.norm(vec)

        x_rand = self.cspace.sample()
        n_sample = len(self.X_sample)
        x_rand_copied = np.repeat(x_rand.reshape(1, -1), n_sample, axis=0)
        sqdists = np.sum((np.vstack(self.X_sample) - x_rand_copied)**2, axis=1)

        idx_nearest = np.argmin(sqdists)
        x_nearest = self.X_sample[idx_nearest]
        if np.linalg.norm(x_rand - x_nearest) > self.eps:
            x_new = x_nearest + unit_vec(x_rand - x_nearest) * self.eps
        else:
            x_new = x_rand

        # update tree
        self.X_sample.append(x_new)
        self.idxes_parents.append(idx_nearest)

    def show(self):
        fig, ax = plt.subplots()
        X_sample_numpy = np.vstack(self.X_sample)
        ax.scatter(X_sample_numpy[:, 0], X_sample_numpy [:, 1], c="black")
        for x, parent_idx in zip(self.X_sample, self.idxes_parents):
            x_parent = self.X_sample[parent_idx]
            ax.plot([x[0], x_parent[0]], [x[1], x_parent[1]], color="red")

if __name__=='__main__':
    b_min = np.zeros(2)
    b_max = np.ones(2)
    cspace = ConfigurationSpace(b_min, b_max)
    rrt = RapidlyExploringRandomTree(cspace, [0.1, 0.1], [0.9, 0.9])
    import time
    ts = time.time()
    for i in range(1000):
        rrt.extend()
    print(time.time() - ts)
