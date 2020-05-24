# https://towardsdatascience.com/how-to-build-your-own-pytorch-neural-network-layer-from-scratch-842144d623f6
# https://towardsdatascience.com/building-neural-network-using-pytorch-84f6e75f9a
import torch
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt
import math
import frmax.mesh as mesh

def show2d(b_min, b_max):
    N_grid = 50
    test_x, _ = mesh.gen_sliced_grid(b_min, b_max, [], [], N_grid)
    #data_ = clf.decision_function(test_x)
    #data = data_.reshape(N_grid, N_grid)

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
        H = 4
        model = nn.Sequential(
                nn.Linear(2, H), 
                nn.Tanh(), 
                nn.Linear(H, H), 
                nn.Linear(H, 1), nn.Sigmoid())
        self.model = model
        self.criterion = nn.BCELoss()

        self.optimizer = optim.SGD(model.parameters(), lr=0.001)
        self.X = nparray2tensor_ifso(X_)
        self.Y = nparray2tensor_ifso(Y_).T

    def optimize(self):
        for i in range(1000):
            Y_pred = self.model(self.X)
            loss = self.criterion(Y_pred, self.Y)
            loss.backward()
            self.optimizer.step()

            lst = list(self.model.parameters())
            print(loss)

    def decision_function(self, X_):
        X = nparray2tensor_ifso(X_)
        Y_pred = self.model(X)
        Y_pred_numpy = Y_pred.detach().numpy().flatten()
        return Y_pred_numpy


def plot_all(X, Y):
    myscat = lambda X, color: plt.scatter(X[:, 0], X[:, 1], c=color)
    myscat(X[Y>0.5], "blue")
    myscat(X[Y<0.5], "red")
    plt.show()



X = np.random.randn(30, 2) * 2
Y = np.array([isInside(x) for x in list(X)])
Y_ = nparray2tensor_ifso(Y)

clf = NNClassifier(X, Y)
clf.optimize()
b_min = np.array([-2., -1.5])
b_max = np.array([2., 1.5])

N_grid = 50
test_x, _ = mesh.gen_sliced_grid(b_min, b_max, [], [], N_grid)
data_ = clf.decision_function(test_x)
data = data_.reshape(N_grid, N_grid)
x_mesh, y_mesh = mesh.gen_mesh(b_min, b_max, N_grid)

fig, ax = plt.subplots()
cs = ax.contourf(x_mesh, y_mesh, data, cmap = 'gray')

myscat = lambda X, color: plt.scatter(X[:, 0], X[:, 1], c=color)
myscat(X[Y>0.5], "blue")
myscat(X[Y<0.5], "red")
plt.show()





"""
m = nn.Sigmoid()
loss = nn.BCELoss()
input = torch.randn(3, requires_grad=True)
target = torch.empty(3).random_(2)
output = loss(m(input), target)
output.backward()
"""
