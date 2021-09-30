import copy
import numpy as np
import pickle
import os
import torch
import torch.nn
import torch.optim
import math
import tqdm

from model import RNNClassifier
from rnn_module import RNNModule

class IshidaLSTM(torch.nn.Module):
    def __init__(self, dim, hid, **kwargs):
        super().__init__()
        rnn = RNNModule(dim, hid, **kwargs)
        self.rnn = rnn
        self.linear = torch.nn.Linear(hid, dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.linear(out)

class IshidaLSTMDataset(torch.utils.data.Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]

class Predictor(object):
    def __init__(self, model):
        self.model = model

    def predict(self, X, n_horizon):
        seq = X
        for i in range(n_horizon):
            seq_pred = self.model(seq)
            x_new = seq_pred[:, -1, :]
            seq = torch.cat((seq, torch.atleast_3d(x_new)), 1)
        return seq

if __name__ == '__main__':
    n_batch = 400
    n_test = 10
    n_seq = 80

    seq = np.linspace(0, math.pi, 10)
    np.sin(seq)

    # generate data
    X_whole = []
    for i in range(n_batch + n_test):
        t_seq = np.linspace(0, 5*math.pi, n_seq)
        x = np.atleast_2d(np.sin(t_seq - np.random.randn()) + np.random.randn(n_seq) * 0.1).T
        X_whole.append(x)
    X_whole = np.stack(X_whole)
    X = X_whole[:n_batch]
    X_test = np.atleast_3d(X_whole[n_batch:])

    import matplotlib.pyplot as plt

    X = torch.from_numpy(X.astype(np.float32))
    X_test = torch.from_numpy(X_test.astype(np.float32))

    mse = torch.nn.MSELoss()
    filename = "model.pickle"
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            model = pickle.load(f)
    else:
        model = IshidaLSTM(1, 200, dropouti=0.5, dropoutw=0.5, dropouto=0.5)
        optim = torch.optim.Adam(model.parameters(), lr=0.001)
        dataset = IshidaLSTMDataset(X)
        train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=10, shuffle=True)

        for i in tqdm.tqdm(range(200)):
            for sample in train_loader:
                optim.zero_grad()
                X_feed = sample[:, :-1, :]
                X_gtruth = sample[:, 1:, :]
                X_pred = model(X_feed)
                loss = mse(X_gtruth, X_pred)
                loss.backward()
                optim.step()
        with open(filename, 'wb') as f:
            pickle.dump(model, f)

    P = Predictor(model)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for i in tqdm.tqdm(range(100)):
        seq = P.predict(X_test[0].view(1, n_seq, 1), 200)
        y = seq.detach().numpy().flatten()
        ax.plot(y, c='red', linewidth=0.3)
    plt.show()
