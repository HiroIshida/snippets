from mimic.augmentation import augment_data, augment_noisy_sequence, randomly_shrink_sequence, randomly_extend_sequence, compute_covariance_matrix
import numpy as np
import torch
import matplotlib.pyplot as plt

def create_random_walk_data(n_seq=30):
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

seqs = create_random_walk_data()
cov = compute_covariance_matrix(seqs)
seq = seqs[0]

seq_shrink = randomly_shrink_sequence(seq)
seq_extend = randomly_extend_sequence(seq, cov)

fig, ax = plt.subplots()

ax.plot(seq[:, 0])
ax.plot(seq_shrink[:, 0])
ax.plot(seq_extend[:, 0])
plt.show()
