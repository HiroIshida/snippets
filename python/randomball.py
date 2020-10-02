import numpy as np
import matplotlib.pyplot as plt
def generate_random_inball(dim, N): 
    """
    this functio generate a random samples inside a hyper-ellipsoid specified by a length vector r of eigen axes. For example, in case of (x/2)^2 + (y/3)^2 = 1, r is r=np.array([2., 3.]).

    Generating a random sample uniformely inside a high dimensional ball is done by

    Barthe, Franck, et al. "A probabilistic approach to the geometry of the $l_{p}^n$-ball." The Annals of Probability 33.2 (2005): 480-513.


    http://mathworld.wolfram.com/BallPointPicking.html
    is wrong. lambda must be 0.5, which means we must set scale in numpy.random.exponetial to be 2.0
    
    """
    y = np.random.exponential(scale=2.0, size=(N))
    X = np.random.randn(dim, N)
    denom = np.sqrt(np.sum(X**2, axis=0)+y)
    rands_ = X/np.tile(denom, (dim, 1))
    return rands_.T

def generate_random_gaussian(dim, N):
    return np.random.multivariate_normal([0]*dim, np.diag([1.0]*dim), N)

def compute_variance(X):
    return sum([np.outer(x, x) for x in X]) / (len(X) - 1)

X = generate_random_inball(2, 3000)
#X = generate_random_gaussian(2, 3000)
C = compute_variance(X)

plt.scatter(X[:, 0], X[:, 1])
