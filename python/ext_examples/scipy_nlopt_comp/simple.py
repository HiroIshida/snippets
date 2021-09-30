from numpy import *
from scipy import optimize
import nlopt 
def scipy_simplest_test():
    print("[TEST]: scipy with SLSQP")
    def f(x):
        return x[0]**2 + x[1]**2
    def jac(x, *args):
        return [2*x[0], 2*x[1]]
    x0 = [2.0, 2.0]
    res = optimize.minimize(f, x0, method="SLSQP", jac=jac, tol=1e-4)
    print(res)

def nlopt_simplest_test(opt_method):
    print("[TEST]: nlopt with {0}".format(opt_method))
    def f(x, grad):
        if grad.size > 0:
            grad[0] = 2 * x[0];
            grad[1] = 2 * x[1];
        return x[0]**2 + x[1]**2
    opt = nlopt.opt(opt_method, 2)
    opt.set_lower_bounds([-10, -10])
    opt.set_min_objective(f)
    opt.set_xtol_rel(1e-4)
    x = opt.optimize([2.0, 2.0])
    minf = opt.last_optimum_value()
    print("optimum at ", x[0], x[1])
    print("minimum value = ", minf)
    print("result code = ", opt.last_optimize_result())

if __name__=='__main__':
    scipy_simplest_test()
    nlopt_simplest_test(nlopt.LD_MMA)
    nlopt_simplest_test(nlopt.LD_SLSQP)

