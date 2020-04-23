# https://towardsdatascience.com/how-to-build-your-own-pytorch-neural-network-layer-from-scratch-842144d623f6
# https://towardsdatascience.com/building-neural-network-using-pytorch-84f6e75f9a
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1)

H = 30
model = nn.Sequential(nn.Linear(1, H),
                      nn.ReLU(),
                      nn.Linear(H, 1))
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
criterion = nn.MSELoss()

func_target = lambda x: 1./(1.+torch.exp(-5*x))
N = 10
inp = torch.randn(N, D_in)
target = func_target(inp)

model.train()
for i in range(1000):
    print(i)
    out = model(inp)
    loss = criterion(out, target)
    print(optimizer.param_groups)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

x_regular = torch.linspace(-2., 2, 200).view(200, 1)
y_real = func_target(x_regular)
y_regular = model(x_regular)

plt.plot(np.array(x_regular.detach()).flatten(), 
        np.array(y_regular.detach()).flatten(), c="b")

plt.plot(np.array(x_regular.detach()).flatten(), 
        np.array(y_real.detach()).flatten(), c="r")

plt.show()

