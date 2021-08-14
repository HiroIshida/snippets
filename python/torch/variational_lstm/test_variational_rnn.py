import copy
import numpy as np
import torch
import torch.nn
import torch.optim
import math

from model import RNNClassifier
from rnn_module import RNNModule

class Predictor(torch.nn.Module):
    def __init__(self, dim, hid, **kwargs):
        super().__init__()
        rnn = RNNModule(dim, hid, **kwargs)
        self.rnn = rnn
        self.linear = torch.nn.Linear(hid, dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.linear(out)

if __name__ == '__main__':
    n_batch = 200
    n_seq = 80

    # generate data
    X = []
    for i in range(n_batch):
        t_seq = np.linspace(0, 4*math.pi, n_seq)
        x = np.atleast_2d(np.sin(t_seq - np.random.randn())).T
        X.append(x)
    X = np.stack(X)

    import matplotlib.pyplot as plt

    X = torch.from_numpy(X.astype(np.float32))

    mse = torch.nn.MSELoss()
    model = Predictor(1, 200, dropouti=0.2, dropoutw=0.4, dropouto=0.3)
    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    for i in range(500):
        optim.zero_grad()
        X_feed = X[:, :-1, :]
        X_gtruth = X[:, 1:, :]
        X_pred = model(X_feed)
        loss = mse(X_gtruth, X_pred)
        print(loss.item())
        loss.backward()
        optim.step()
