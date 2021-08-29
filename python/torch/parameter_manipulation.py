import torch
import numpy as np

y = torch.from_numpy(np.array([[0, 1], [1, 0], [0, 1]])).float()

def loop_case():
    loss_list = []
    p = torch.nn.Parameter(torch.zeros(2))
    for i in range(3):
        # In the loop, gradient is just accumulated
        loss = torch.nn.MSELoss(reduction='sum')(y[i, :], p)
        loss_list.append(loss)
        loss.backward()
    print(p.grad)

def repeat_case():
    p = torch.nn.Parameter(torch.zeros(2))
    # P share the smae p through the repeated components
    P = p.repeat(3, 1)
    loss = torch.nn.MSELoss(reduction='sum')(y, P)
    loss.backward()
    print(p.grad)

loop_case()
repeat_case()
