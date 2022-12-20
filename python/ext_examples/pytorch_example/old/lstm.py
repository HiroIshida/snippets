import numpy as np
import torch
from torch import nn
import torch.optim as optim
import tqdm

class SinWaveDataset(torch.utils.data.Dataset):
    def __init__(self):
        X_np = np.linspace(0, 12.0, 2000)
        Y_np = np.sin(X_np)
        self._X = torch.from_numpy(X_np)
        self._Y = torch.from_numpy(Y_np)
        self._window_size = 50

    def __len__(self):
        size = len(self._Y) - self._window_size + 1 - 1
        return size

    def __getitem__(self, idx):
        assert idx < self.__len__()
        n_dim = 1
        feed = self._Y[idx:idx+self._window_size].view(self._window_size, n_dim).float()
        pred = self._Y[idx+self._window_size].view(n_dim).float()
        return feed, pred

class MyLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self._n_hid = 10
        self._lstm = nn.LSTM(1, self._n_hid, batch_first=True)
        self._linear = nn.Linear(self._n_hid, 1)
        self._sigmoid = nn.Sigmoid()

    def forward(self, x_seq):
        x_seq = x_seq.float()
        n_batch, n_seq, n_dim = x_seq.shape
        out, _ = self._lstm(x_seq, None)
        final = self._linear(out[:, -1, :]).view(-1, n_dim)
        return final

if __name__=='__main__':
    dataset = SinWaveDataset()
    n_data = len(dataset)
    n_train = round(n_data * 0.9)
    n_val = n_data  - n_train
    train_set, val_set =  torch.utils.data.random_split(dataset, [n_train, n_val])
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=100, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=100, shuffle=False)

    lstm = MyLSTM()
    fn_loss = nn.MSELoss()
    optimizer = optim.Adam(lstm.parameters(), lr=0.01)

    for epoch in range(1000):
        lstm.train()
        train_loss = 0.0
        for feed, pred_truth in train_loader:
            optimizer.zero_grad()
            pred = lstm(feed)
            loss = fn_loss(pred, pred_truth)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss/len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

