import torch

# Define two data points
x1 = torch.tensor([1., 1])
y1 = torch.tensor([-2.8])

x2 = torch.tensor([2., 1])
y2 = torch.tensor([7.2])

# Now consider regression problem with y = A * x + b
A = torch.ones(2, requires_grad=True)
b_list = [torch.ones(1, requires_grad=True) for _ in range(2)]

loss1 = torch.nn.MSELoss()(y1, torch.dot(A, x1) + b_list[0])
loss1.backward()
print(A.grad)
print([b.grad for b in b_list])

loss2 = torch.nn.MSELoss()(y2, torch.dot(A, x2) + b_list[1])
loss2.backward()
print(A.grad)
print([b.grad for b in b_list])
