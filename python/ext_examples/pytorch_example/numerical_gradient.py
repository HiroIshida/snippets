import copy
from torch.nn import Module
import torch.nn as nn 
import torch

torch.manual_seed(0)

if __name__ == "__main__":
    # take away: using torch.float32 will result in a large numerical error
    dtype = torch.float64
    x_list = torch.stack([torch.randn(3, dtype=dtype) for _ in range(100)])
    y_list = torch.stack([torch.randn(1, dtype=dtype) for _ in range(100)])
    linear_model = nn.Linear(3, 1, dtype=dtype)

    # gradient using torch's autograd
    loss = nn.MSELoss()(linear_model(x_list), y_list)
    loss.backward()
    grad_as_vector = linear_model.weight.grad.view(-1)
    print(f"torch autograd: {grad_as_vector}")

    # gradient using numerical differentiation
    grad_numel = torch.zeros_like(grad_as_vector)
    delta = 1e-6
    for i in range(3):
        model_plus = copy.deepcopy(linear_model)
        model_plus.weight.data[0][i] += delta
        loss_plus = nn.MSELoss()(model_plus(x_list), y_list)
        grad_numel[i] = (loss_plus - loss) / delta
    print(f"numerical gradient: {grad_numel}")
