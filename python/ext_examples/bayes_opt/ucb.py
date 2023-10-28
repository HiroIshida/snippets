import numpy as np
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization, UtilityFunction
np.random.seed(0)

pbounds = {'x': (-3, 1.5)}

def black_box_function(x):
    return 1/(2*np.pi)**0.5*np.exp(-0.5*x**2)

optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    random_state=1,
)


utility = UtilityFunction()

for i in range(10):
    if i > 0:
        next_point = optimizer.suggest(utility)
    else:
        next_point = {"x": -2.5}
        # next_point = optimizer.suggest(utility)

    print(next_point)
    target = black_box_function(**next_point)
    optimizer.register(params=next_point, target=target)

    gp = optimizer._gp
    plt.figure(figsize=(12, 5))
    xlin = np.linspace(-3, 3, 1000)
    plt.plot(xlin, black_box_function(xlin), 'r:', label='Objective Function')
    plt.plot(optimizer.space.params.flatten(), optimizer.space.target, 'r.', markersize=10, label='Observations')
    mu, sigma = gp.predict(xlin.reshape(-1, 1), return_std=True)
    plt.plot(xlin, mu, 'b-', label=r'$\mu(x)$')
    plt.fill_between(xlin, mu-2*sigma, mu+2*sigma, color='b', alpha=0.2, label=r'$2\sigma(x)$')
    plt.legend()
    plt.show()
