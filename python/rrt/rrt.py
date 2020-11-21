import numpy as np
import matplotlib.pyplot as plt 

#np.random.seed(0)

class ConfigurationSpace(object):
    def __init__(self, b_min, b_max):
        self.b_min = b_min
        self.b_max = b_max
        self.n_dof = len(b_min)

    def sample(self):
        w = self.b_max - self.b_min
        return np.random.rand(self.n_dof) * w + self.b_min

class RapidlyExploringRandomTree(object): 
    def __init__(self, cspace, q_start, pred_goal_condition=None, pred_valid_config=None,
            N_maxiter=30000):
        self.cspace = cspace
        self.eps = 0.05
        self.n_resolution = 10
        self.N_maxiter = N_maxiter

        if pred_goal_condition is None:
            pred_goal_condition = lambda q: False

        if pred_valid_config is None:
            pred_valid_config = lambda q : True

        self.isValid = pred_valid_config
        self.isGoal = pred_goal_condition

        # reserving memory is the key 
        self._Q_sample = np.zeros((N_maxiter, cspace.n_dof))
        self._idxes_parents = np.zeros(N_maxiter, dtype='int64')

        # set initial sample
        self.n_sample = 1
        self._Q_sample[0] = q_start
        self._idxes_parents[0] = 0 # self reference

    @property
    def q_start(self):
        return self.Q_sample[0]

    @property
    def Q_sample(self): # safer 
        return self._Q_sample[:self.n_sample]

    @property
    def idxes_parents(self): # safer 
        return self._idxes_parents[:self.n_sample]

    def extend(self):
        if self.n_sample ==  self.N_maxiter:
            raise Exception

        def unit_vec(vec):
            return vec/np.linalg.norm(vec)

        q_rand = self.cspace.sample()
        sqdists = np.sum((self.Q_sample - q_rand[None, :])**2, axis=1)
        idx_nearest = np.argmin(sqdists)
        q_nearest = self.Q_sample[idx_nearest]
        if np.linalg.norm(q_rand - q_nearest) > self.eps:
            q_new = q_nearest + unit_vec(q_rand - q_nearest) * self.eps
        else:
            q_new = q_rand

        # update tree
        q_new_reshaped = q_new.reshape(1, -1)
        if self.isValid(q_new_reshaped):
            self._Q_sample[self.n_sample] = q_new
            self._idxes_parents[self.n_sample] = idx_nearest
            self.n_sample += 1
            return self.isGoal(q_new)
        return False # not goal

    def backward_path(self, idx_end):
        q_seq = [self.Q_sample[idx_end]]
        idx_parent = self.idxes_parents[idx_end]
        idx_root = 0
        while idx_parent != idx_root:
            q_seq.append(self.Q_sample[idx_parent])
            idx_parent = self.idxes_parents[idx_parent]
        q_seq.append(self.q_start)
        return q_seq

    def show(self, fax=None):
        if fax is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = fax
        n = self.n_sample
        ax.scatter(self.Q_sample[:, 0], self.Q_sample[:, 1], c="black", s=5)
        for q, parent_idx in zip(self.Q_sample, self.idxes_parents):
            q_parent = self.Q_sample[parent_idx]
            ax.plot([q[0], q_parent[0]], [q[1], q_parent[1]], color="red", linewidth=0.5)

class BidirectionalRRT(object):
    def __init__(self, cspace, q_start, q_goal, pred_valid_config,
            N_maxiter=10000):
        self.rrt1 = RapidlyExploringRandomTree(cspace, q_start, pred_valid_config=pred_valid_config)
        self.rrt2 = RapidlyExploringRandomTree(cspace, q_goal, pred_valid_config=pred_valid_config)

        self.connect_pair = None # [idx_of_rrt1, idx_or_rrt2]
        self.solution = None

        # because setting it is mutual referencial, we can set 
        # pred_goal_condition only after initialization of rrt1 and rrt2
        def goalpred_for_rrt1(q):
            diffs = self.rrt2.Q_sample - q[None, :]
            sqdists = np.sum(diffs ** 2, axis=1)
            idx_min = np.argmin(sqdists)
            if sqdists[idx_min] < self.rrt1.eps ** 2:
                idx_self = self.rrt1.n_sample - 1
                idx_other = idx_min
                self.connect_pair = [idx_self, idx_other]
                return True
            return False

        self.rrt1.isGoal = goalpred_for_rrt1 # overwrite!

    def solve(self):
        while self.connect_pair is None:
            self.rrt2.extend()
            self.rrt1.extend()
        print("solved")

        sol_forward = self.rrt1.backward_path(self.connect_pair[0])
        sol_forward.reverse()
        sol_backward = self.rrt2.backward_path(self.connect_pair[1])
        solution = sol_forward + sol_backward
        self.solution = solution
        return solution

    def show(self, fax=None):
        if fax is None:
            fax = plt.subplots()
        self.rrt1.show(fax)
        self.rrt2.show(fax)
        fig, ax = fax
        if self.solution is not None:
            for i in range(len(self.solution) - 1):
                x = [self.solution[i][0], self.solution[i+1][0]]
                y = [self.solution[i][1], self.solution[i+1][1]]
                ax.plot(x, y, color="green")

            # show connected pair
            p1 = self.rrt1.Q_sample[self.connect_pair[0]]
            p2 = self.rrt2.Q_sample[self.connect_pair[1]]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="blue")
        return fig, ax


if __name__=='__main__':
    b_min = np.zeros(2)
    b_max = np.ones(2)
    cspace = ConfigurationSpace(b_min, b_max)
    q_start = np.array([0.1, 0.1])
    q_goal = np.array([0.9, 0.9])
    sdf = lambda q: np.linalg.norm(q - np.array([0.5, 0.5])) - 0.3
    pred_valid_config = lambda q: sdf(q) > 0.0

    brrt = BidirectionalRRT(cspace, q_start, q_goal, pred_valid_config)
    brrt.solve()
    fig, ax = brrt.show()
    ax.axis("equal")
    plt.show()
