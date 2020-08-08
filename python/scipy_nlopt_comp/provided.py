import numpy as np
from scipy import optimize
from scipy.optimize import NonlinearConstraint, Bounds
from math import *
import nlopt 
def scipy_provided_test(x0):
    print("[TEST]: scipy with SLSQP starting")
    def f(x):
        return sqrt(x[0])
    def gen_const(a, b):
        h = lambda x: (a*x[0] + b)**3 - x[1]
        nlc = NonlinearConstraint(h, 0, np.inf)
        return nlc
    try:
        res = optimize.minimize(f, x0, method="SLSQP", tol=1e-4,
                bounds=(Bounds(0, np.inf)),
                constraints=(gen_const(2., 0.), gen_const(-1., 1.)))
        print(res)
    except Exception as e:
        print(str(e))

def nlopt_provided_test(init_solution):
    print("[TEST]: nlopt with SLSQP starting from {0}".format(init_solution))
    def myfunc(x, grad):
        if grad.size > 0:
            grad[0] = 0.0
            grad[1] = 0.5 / sqrt(x[1])
        return sqrt(x[1])
    def myconstraint(x, grad, a, b):
        if grad.size > 0:
            grad[0] = 3 * a * (a*x[0] + b)**2
            grad[1] = -1.0
        return (a*x[0] + b)**3 - x[1]
    opt = nlopt.opt(nlopt.LD_SLSQP, 2)
    opt.set_lower_bounds([-float('inf'), 0])
    opt.set_min_objective(myfunc)
    opt.add_inequality_constraint(lambda x,grad: myconstraint(x,grad,2,0), 1e-8)
    opt.add_inequality_constraint(lambda x,grad: myconstraint(x,grad,-1,1), 1e-8)
    opt.set_xtol_rel(1e-4)
    try:
        x = opt.optimize(init_solution)
        minf = opt.last_optimum_value()
        print("optimum at ", x[0], x[1])
        print("minimum value = ", minf)
        print("result code = ", opt.last_optimize_result())

    except Exception as e:
        print(str(e))

scipy_provided_test([0.0, 5.0])
scipy_provided_test([1.234, 5.678])
nlopt_provided_test([0.0, 5.0]) 
nlopt_provided_test([1.234, 5.678]) # provided example

