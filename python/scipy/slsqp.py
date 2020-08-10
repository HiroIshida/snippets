import numpy as np
from scipy.optimize import *

bounds = Bounds([0, -0.5], [1.0, 2.0])

def ineq_fun(x):
    ret = np.array([1 - x[0] - 2*x[1],
             1 - x[0]**2 - x[1],
             1 - x[0]**2 + x[1]])
    return ret

def ineq_jac_fun(x):
    ret = np.array([[-1.0, -2.0],
             [-2*x[0], -1.0],
             [-2*x[0], 1.0]])
    return ret

ineq_cons = {'type': 'ineq',
             'fun' : ineq_fun,
             'jac' : ineq_jac_fun}
eq_cons = {'type': 'eq',
           'fun' : lambda x: np.array([2*x[0] + x[1] - 1]),
           'jac' : lambda x: np.array([2.0, 1.0])}

x0 = np.array([0.5, 0])
res = minimize(rosen, x0, method='SLSQP', jac=rosen_der,
               constraints=[eq_cons, ineq_cons], options={'ftol': 1e-9, 'disp': True},
               bounds=bounds)

print(res.x)
