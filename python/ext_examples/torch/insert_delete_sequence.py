import torch
import numpy as np
import matplotlib.pyplot as plt
from traj_augmentation import create_random_walk_data
from traj_augmentation import compute_covariance_matrix

def delete_random_index(seq: torch.Tensor) -> torch.Tensor:
    # choose delete index
    n_seq_len, _ = seq.shape
    idxes_delete = np.random.randint(n_seq_len, size=int(n_seq_len * np.random.rand() * 0.5))
    idxes_nondelete = set(np.arange(n_seq_len)) - set(idxes_delete)
    return seq[list(idxes_nondelete)]

def insert_random_index(seq: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
    # insertion
    n_seq_len, n_dim = seq.shape
    idxes_insert = np.random.randint(n_seq_len, size=int(n_seq_len * np.random.rand() * 0.5))

    seq_new = []
    for i in range(n_seq_len):
        seq_new.append(seq[i]) 
        if i in set(idxes_insert):
            n_insert_len = np.random.randint(int(n_seq_len * 0.2))
            noises = np.random.multivariate_normal(mean=np.zeros(n_dim), cov=cov, size=n_insert_len)
            for j in range(n_insert_len):
                seq_new.append(seq[i] + noises[j])
    return torch.stack(seq_new)

seqs = create_random_walk_data(n_seq=30)
seq = seqs[0]
cov = compute_covariance_matrix(seqs)

fig, ax = plt.subplots()

a = delete_random_index(seq)
print(a.shape)
b = insert_random_index(a, cov)
print(b.shape)

ax.plot(seq[:, 0])
ax.plot(a[:, 0])
ax.plot(b[:, 0])
plt.show()
