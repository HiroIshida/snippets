import matplotlib.pyplot as plt
import numpy as np
import reeds_shepp
import json
q0 = [0.0, 0.0, 0.]
q1 = np.random.randn(3) * 0.2
q1[2] += 1

eps = 0.0001
q_seq = np.array(reeds_shepp.path_sample(q0, q1, 0.2, eps))[:, :3]
q_seq_back = np.array(reeds_shepp.path_sample(q1, q0, 0.2, eps))[:, :3]

a = len(q_seq)
b = len(q_seq_back)

plt.scatter(q_seq[:, 0], q_seq[:, 1], c='r', s=0.01)
plt.scatter(q_seq_back[:, 0], q_seq_back[:, 1], c='b', s=0.01)
plt.show()


