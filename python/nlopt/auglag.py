import numpy as np
import nlopt
import copy  

ndim = 2
algorithm = nlopt.LD_SLSQP
#algorithm = nlopt.LD_SLSQP
#algorithm = nlopt.LN_AUGLAG
opt = nlopt.opt(algorithm, ndim)

def auto_diff(f, x0):
    dim = len(x0)
    dx = 1e-6
    f0 = f(x0)
    grad = np.zeros(dim)
    for i in range(dim):
        x_ = copy.copy(x0)
        x_[i] += dx
        grad[i] = (f(x_) - f(x0))/dx
    return grad


def func(x, grad):
    f = lambda x: np.linalg.norm(x) ** 2
    val = f(x)
    if grad.size > 0:
        for i in range(2):
            diff = auto_diff(f, x)
            grad[i] = diff[i]
    return val

def fc(x, grad):
    f = lambda x: x[0] * x[1] - 1
    if grad.size > 0:
        for i in range(2):
            diff = auto_diff(f, x)
            grad[i] = diff[i]

tol = 1e-1 
opt.set_ftol_rel(tol)
opt.set_min_objective(func)
#opt.add_equality_constraint(fc, tol=1e-10)
x_init_guess = np.array([1.0, 1.0])
xopt = opt.optimize(x_init_guess)
