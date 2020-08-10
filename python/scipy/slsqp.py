import numpy as np
from scipy.optimize import *

global jac_cache
jac_cache = None

def ineq_fun(x):
    fx = np.array([1 - x[0] - 2*x[1], 1 - x[0]**2 - x[1], 1 - x[0]**2 + x[1]])
    global jac_cache
    jac_cache = np.array([[-1.0, -2.0], [-2*x[0], -1.0], [-2*x[0], 1.0]])
    return fx

def ineq_jac_fun(x):
    global jac_cache
    if jac_cache is None:
        raise Exception
    return jac_cache

ineq_cons = {'type': 'ineq',
             'fun' : ineq_fun,
             'jac' : ineq_jac_fun}
eq_cons = {'type': 'eq',
           'fun' : lambda x: np.array([2*x[0] + x[1] - 1]),
           'jac' : lambda x: np.array([2.0, 1.0])}

bounds = Bounds([0, -0.5], [1.0, 2.0])
x0 = np.array([0.5, 0])
res = minimize(rosen, x0, method='SLSQP', jac=rosen_der,
               constraints=[eq_cons, ineq_cons], options={'ftol': 1e-9, 'disp': True},
               bounds=bounds)

print(res.x)
