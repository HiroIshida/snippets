import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import UnwhitenedVariationalStrategy
import numpy as np
import torch

kern = gpytorch.kernels.RBFKernel()
covar = gpytorch.kernels.ScaleKernel(kern)
X1 = torch.randn(1000, 2)
X2 = torch.randn(1000, 2)

with torch.no_grad():
    k = covar(X1, X2)

