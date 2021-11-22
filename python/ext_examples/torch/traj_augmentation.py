from typing import List

import numpy as np
import torch

def create_random_walk_data(n_seq=100):
    walks = []
    for _ in range(n_seq):
        seq = []
        x = np.random.randn(3)
        seq.append(x)
        for _ in range(100 + np.random.randint(10)):
            x = x + np.random.randn(3) * [0.1, 0.2, 0.3]
            seq.append(x)
        seq_torch = torch.from_numpy(np.array(seq))
        walks.append(seq_torch)
    return walks

def compute_covariance_matrix(walks):
    diffs = []
    for walk in walks:
        x_pre = walk[:-1, :]
        x_post = walk[1:, :]
        diff = x_post - x_pre
        diffs.append(diff)
    diffs_cat = torch.cat(diffs, axis=0)
    cov = torch.cov(diffs_cat.T)
    return cov

def augment_data(walks: List[torch.Tensor], n_aug=10, cov_scale=1.0):
    cov = compute_covariance_matrix(walks) * cov_scale ** 2
    cov_dim = cov.shape[0]
    walks_new = []
    for walk in walks:
        n_seq, n_dim = walk.shape
        assert cov_dim == n_dim
        for _ in range(n_aug):
            rand_aug = np.random.multivariate_normal(mean=np.zeros(n_dim), cov=cov, size=n_seq)
            assert rand_aug.shape == walk.shape
            walks_new.append(walk + torch.from_numpy(rand_aug).float())
    return walks_new

walks = create_random_walk_data(1000)
cov = compute_covariance_matrix(walks)
walks_new = augment_data(walks, n_aug=30)
cov_new = compute_covariance_matrix(walks_new)
print(cov)
print(cov_new)


