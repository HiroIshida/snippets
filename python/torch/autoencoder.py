import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import tqdm
import pickle

DEVICE = 'cpu'
SEED = 0
CLASS_SIZE = 10
BATCH_SIZE = 256
ZDIM = 16
NUM_EPOCHS = 25

# Set seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)   
torch.cuda.manual_seed(SEED)

_default_cache_dir = os.path.expanduser('~/.torchvision')

# Train
train_dataset = torchvision.datasets.MNIST(
    root=_default_cache_dir,
    train=True,
    transform=transforms.ToTensor(),
    download=True,
)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class ConvAutoEncoder(nn.Module):
    def __init__(self, n_batch):
        # Saito, Namiko, et al. "How to Select and Use Tools?: Active Perception of Target Objects Using Multimodal Deep Learning." IEEE Robotics and Automation Letters 6.2 (2021): 2517-2524.
        self.n_batch = n_batch

        super().__init__()

        self._encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, stride=(2, 2)), # 16x14x14
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1, stride=(2, 2)), # 32x7x7
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, stride=(2, 2)), # 64x4x4
            nn.ReLU(inplace=True), # 64x4x4
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 16)
        )
        # https://blog.shikoan.com/pytorch-convtranspose2d/
        self._decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.Linear(64, 1024),
            nn.ReLU(inplace=True),
            Reshape(-1, 64, 4, 4),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
            )

    def forward(self, x):
        outs = model._decoder(model._encoder(x))
        return outs

def save_pickle(model, filename="nnmodel.pickle"):
    with open(filename, "wb") as f:
        pickle.dump(model, f)

if __name__=='__main__':
    fn_loss = nn.BCELoss()
    images, labels = train_loader.__iter__().__next__()
    model = ConvAutoEncoder(256)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    #Epochs
    n_epochs = 20

    for epoch in range(1, n_epochs+1):
        save_pickle(model)
        model.train()
        # monitor training loss
        train_loss = 0.0

        #Training
        for images, labels in tqdm.tqdm(train_loader):
            images = images.to("cpu")
            optimizer.zero_grad()
            outputs = model(images)
            loss = fn_loss(outputs, images)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*images.size(0)
              
        train_loss = train_loss/len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

