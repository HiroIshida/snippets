# https://github.com/CyberAgentAILab/cmaes
import numpy as np
from cmaes import CMA


def quadratic(x) -> float:
    f = np.sum((x - 1)**2)
    g = max(sum(x ** 2) - 1, 0.0)
    return f + 10000 * g

if __name__ == "__main__":
    n_dim = 20
    optimizer = CMA(mean=np.zeros(n_dim), sigma=3.0)

    for generation in range(1000):
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = quadratic(x)
            solutions.append((x, value))
            print(x)
        optimizer.tell(solutions)

    print("actual optima: {}".format(np.ones(n_dim) / np.sqrt(n_dim)))
