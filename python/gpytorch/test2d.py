import torch
import gpytorch
import math
from matplotlib import cm
from matplotlib import pyplot as plt
import numpy as np

# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.likelihood = likelihood
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def optimize(self):
        self.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam([
            {'params': self.parameters()},  # Includes GaussianLikelihood parameters
        ], lr=0.05)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        try:
            for i in range(10):
                print("hoge")
                optimizer.zero_grad()
                output = self(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.step()
        except RuntimeError:
            print("cannot be optimized, more training data is necessary pribably")

        # switch to eval mode
        self.eval()
        self.likelihood.eval()

    def predict(self, test_x):
        with torch.no_grad():
            pred = self.likelihood(model(test_x))
            return (pred.mean, pred.variance)



import torch as th
#X_init = th.tensor([[0.5, 0.3], [0.5, 0.5], [0.5, 0.7]])
#Y_init = th.tensor([1, -1, 1])
def gen_dataset(N):
    predicate = lambda x: (x[0]**2 -0.5)**2 + (x[1]**2 -0.5)**2 < 0.4 ** 2
    X = []
    Y = []
    for i in range(N):
        x = th.rand(2)  
        y = predicate(x) * 2 - 1
        X.append(x)
        Y.append(y)
    X_ = th.stack(X, dim=0)
    Y_ = th.stack(Y)
    return X_, Y_
X_init, Y_init = gen_dataset(30)


likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(X_init, Y_init, likelihood)
model.optimize()

#model.predict(torch.tensor([[0.5, 0.5]]))

# Initialize plots
fig, ax = plt.subplots(1, 1, figsize=(14, 10))

# Test points
n1, n2 = 50, 50
xv, yv = torch.meshgrid([torch.linspace(0, 1, n1), torch.linspace(0, 1, n2)])

# Make predictions
with torch.no_grad(), gpytorch.settings.fast_computations(log_prob=False, covar_root_decomposition=False):
    test_x = torch.stack([xv.reshape(n1*n2, 1), yv.reshape(n1*n2, 1)], -1).squeeze(1)
    predictions = likelihood(model(test_x))
    mean = predictions.mean

extent = (xv.min(), xv.max(), yv.max(), yv.min())
ax.imshow(mean.detach().numpy().reshape(n1, n2), extent=extent, cmap=cm.jet)
plt.show()

