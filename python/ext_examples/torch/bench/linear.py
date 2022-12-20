import torch.nn as nn
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import threadpoolctl

def measure_perf(depth, with_grad: bool = False):
    lst = []
    for _ in range(depth):
        lst.append(nn.Linear(40, 40))
        lst.append(nn.ReLU())
    lst.append(nn.Linear(40, 1))
    lst.append(nn.Sigmoid())
    net = nn.Sequential(*lst)

    arr = np.random.randn(1, 40)
    ten = torch.from_numpy(arr).float()

    ten.requires_grad_(with_grad)

    ts = time.time()
    n_trial = 100
    for _ in range(n_trial):
        val1 = net(ten)
        if with_grad:
            val1.backward()
    perf = (time.time() - ts) / n_trial
    return perf


perfs = [measure_perf(n, True) for n in tqdm.tqdm(range(50))]
plt.plot(perfs)
plt.show()

