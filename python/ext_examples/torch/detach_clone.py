# show that without detach().clone(), requires heap memory a lot in vain
import torch

N = 3000000
w = torch.randn(N)
w.requires_grad = True

loss_list = []
for _ in range(1000):
    a = torch.randn(N)
    a.requires_grad = True
    loss = torch.dot(a, w)

    # with only detach() all the huge computational graph is copied
    # with only clone() also the huge computational graph is copied
    # so we need detach().clone()
    loss_list.append(loss.clone())
