import numpy as np
import torch
from torch import nn
from math import pi

def strange_wave_data():
    X = np.linspace(0, pi * 6, 200)
    Y = np.hstack((np.sin(X)[:-1], np.sin(X) * 2))
    return Y.reshape(1, -1, 1)

def PhasedSequenceGen(X, splits):
    n = len(splits)+1
    for i in range(n):
        if i==0:
            yield X[:, :splits[i], :], X[:, 1:splits[i]+1, :]
        elif i==n-1:
            yield X[:, splits[i-1]:-1, :], X[:, splits[i-1]+1:, :]
        else:
            yield X[:, splits[i-1]:splits[i], :], X[:, splits[i-1]+1:splits[i]+1, :]

class PBRNN(nn.Module):
    def __init__(self, n_dim, n_pb, n_hid=200):
        super().__init__()
        self._n_hid = n_hid
        self._lstm = nn.LSTM(n_dim, self._n_hid, batch_first=True)
        self._linear = nn.Linear(self._n_hid, n_dim)
        self._sigmoid = nn.Sigmoid()

    def forward(self, X):
        # In order to use as a seq-first data we do like
        X = X.permute(1, 0, 2)

        n_seq, n_batch, n_dim = X.shape
        hc_tuple = None
        for i in range(n_seq):
            x = X[None, i, :, :]
            _, hc_tuple = self._lstm(x, hc_tuple)

    def loss(self, G):
        # https://discuss.pytorch.org/t/what-exactly-does-retain-variables-true-in-loss-backward-do/3508/25
        hc_tuple = None
        for x, x_pred_gt in G:
            out, hc_tuple = self._lstm(x, hc_tuple)
            x_pred = self._linear(out)
            length = max(x.shape)
            loss = nn.MSELoss()(x_pred, x_pred_gt) * length
            loss.backward(retain_graph=True)

if __name__=='__main__':
    X = torch.from_numpy(strange_wave_data()).float()
    G = PhasedSequenceGen(X, [200])
    model = PBRNN(1)
    model.loss(G)

