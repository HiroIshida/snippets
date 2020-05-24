# https://towardsdatascience.com/how-to-build-your-own-pytorch-neural-network-layer-from-scratch-842144d623f6
# https://towardsdatascience.com/building-neural-network-using-pytorch-84f6e75f9a
import torch
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt
import math

#torch.manual_seed(1)

def isInside(vec):
    x, y = vec
    y_ = math.exp(-x**2/2)
    return abs(y) < y_

def nparray2nested_ifso(x):
    isNested = len(x.shape) > 1
    if isNested:
        return x
    else:
        return np.array([x])

def nparray2tensor_ifso(x):
    if not isinstance(x, np.ndarray):
        return x
    x_ = nparray2nested_ifso(x)
    tensor_ = torch.from_numpy(x_)
    tensor = tensor_.float()
    return tensor

class NNClassifier:
    def __init__(self, X_, Y_):
        H = 30
        model = nn.Sequential(nn.Linear(2, H), nn.Tanh(), nn.Linear(H, 1), nn.Sigmoid())
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.1)
        self.X = nparray2tensor_ifso(X_)
        self.Y = nparray2tensor_ifso(Y_).T

    def optimize(self):
        for i in range(1000):
            Y_pred = self.model(self.X)
            #loss = self.criterion(Y_pred, self.Y.flatten())
            #loss.backward()
            #self.optimizer.step()


X = np.random.randn(1000, 2) * 2
Y = np.array([isInside(x) for x in list(X)])

nnc = NNClassifier(X, Y)
nnc.optimize()
#Y_pred = nnc.model(nparray2tensor_ifso(X))

"""
myscat = lambda X, color: plt.scatter(X[:, 0], X[:, 1], c=color)
myscat(X[Y>0.5], "blue")
myscat(X[Y<0.5], "red")
plt.show()
"""





