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
    def __init__(self, state_dim, pb_dim, n_hid=200):
        super().__init__()
        self.pb_dim = pb_dim
        self.n_hid = n_hid

        state_aug_dim = state_dim + pb_dim

        self._lstm = nn.LSTM(state_aug_dim, self.n_hid, batch_first=True)
        self._linear = nn.Linear(self.n_hid, state_dim)
        self._sigmoid = nn.Sigmoid()

        self._parametric_bias = torch.nn.Parameter(torch.zeros(pb_dim))

    def forward(self, X):
        # In order to use as a seq-first data we do like
        X = X.permute(1, 0, 2)

        n_seq, n_batch, n_dim = X.shape
        hc_tuple = None
        for i in range(n_seq):
            x = X[None, i, :, :]
            _, hc_tuple = self._lstm(x, hc_tuple)

    def loss(self, G):
        # as for retain_graph=True
        # https://discuss.pytorch.org/t/what-exactly-does-retain-variables-true-in-loss-backward-do/3508/25

        # as for repeat parameter
        # https://discuss.pytorch.org/t/repeat-a-nn-parameter-for-efficient-computation/25659
        hc_tuple = None
        for x, x_pred_gt in G:
            seq_length = max(x.shape)
            tmp = self._parametric_bias.repeat(seq_length, 1)
            x_aug = torch.cat((x, tmp[None, :, :]), 2) # augmented with pb
            out, hc_tuple = self._lstm(x_aug, hc_tuple)

            x_pred = self._linear(out)
            loss = nn.MSELoss()(x_pred, x_pred_gt) * seq_length
            loss.backward(retain_graph=True)

if __name__=='__main__':
    X = torch.from_numpy(strange_wave_data()).float()
    G = PhasedSequenceGen(X, [200])
    model = PBRNN(1, 3)
    model.loss(G)

