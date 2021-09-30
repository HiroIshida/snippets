import torch
import gpytorch
from matplotlib import pyplot as plt
from matplotlib import cm

from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import UnwhitenedVariationalStrategy

class GPClassificationModel(ApproximateGP):
    def __init__(self, train_x, train_y):
        variational_distribution = CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = UnwhitenedVariationalStrategy(
            self, train_x, variational_distribution, learn_inducing_locations=False
        )
        super(GPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        kern = gpytorch.kernels.RBFKernel(ard_num_dims=2)
        kern.lengthscale = torch.tensor([0.5, 0.1])
        self.covar_module = gpytorch.kernels.ScaleKernel(kern)
        self.likelihood = gpytorch.likelihoods.BernoulliLikelihood()

        self.train()
        self.likelihood.train()

        def custom_auto_tune_params():
            """ this is how we can remove certain params from optimization """
            for param in self.named_parameters():
                name = param[0]
                if name != "covar_module.base_kernel.raw_lengthscale":
                    yield param[1]

        optimizer = torch.optim.Adam(custom_auto_tune_params(), lr=0.1)

        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self, train_y.numel())

        for i in range(30):
            optimizer.zero_grad()
            output = self.__call__(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

        self.eval()
        self.likelihood.eval()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred

if __name__=='__main__':
    # Initialize model and likelihood
    train_x = torch.tensor([[0.5, 0.2], [0.6, 0.5], [0.5, 0.9], [0.4, 0.5], [0.3, 0.9]])
    train_y = torch.tensor([0, 1, 0, 1, 0])

    model = GPClassificationModel(train_x, train_y)

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    N = 100
    import numpy as np
    xlin = ylin = np.linspace(0, 1, N)
    X, Y = np.meshgrid(xlin, ylin)
    pts = torch.tensor([[x,y] for x, y in list(zip(X.flatten(), Y.flatten()))]).float()

    with torch.no_grad(), gpytorch.settings.fast_computations(log_prob=False, covar_root_decomposition=False):
        predictions = model.likelihood(model(pts))
        mean = predictions.mean.detach().numpy()
    ax.contourf(X, Y, mean.reshape((N, N)))
    plt.show()
