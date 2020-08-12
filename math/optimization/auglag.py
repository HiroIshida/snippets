import copy
import numpy as np 
import scipy.optimize as opt

def compute_grad(f, x):
    dim = len(x)
    eps = 1e-7
    f0 = f(x)
    grad = np.zeros(dim)
    for i in range(dim): 
        x1 = copy.copy(x)
        x1[i] += eps
        f1 = f(x1)
        grad[i] = (f1 - f0)/eps
    return grad

def check_kkt_condition(x_sol, lam_sol, f, ceq): 
    print(x_sol)
    Lag = lambda x: f(x) - lam_sol * ceq(x)
    grad_Lag = compute_grad(Lag, x_sol)

    eps = 1e-4
    assert np.linalg.norm(grad_Lag) < eps, "grad of lag is {0}".format(grad_Lag)
    print("passed condition of laggrad")
    assert ceq(x_sol) < eps
    print("passed feasiblity")

def auglag(x0, lam0, mu, tau, f, ceq):
    x_now = x0
    lam_now = lam0
    mu_now = mu
    tau_now = tau

    for i in range(50):
        Lag = lambda x: f(x) - lam_now * ceq(x) + 1.0/(2 * mu_now) * ceq(x)**2
        res = opt.minimize(Lag, x_now, method='BFGS', options={'disp': False, 'gtol': tau})
        x_star_approx = res.x

        x_now = x_star_approx
        lam_now -= ceq(x_star_approx)/mu_now
        mu_now *= 0.9 # if I set it to smaller, the x_opt diverges ...
        tau_now *= 0.1

    check_kkt_condition(x_now, lam_now, f, ceq)
    return x_now

f = lambda x : x[0] ** 2 + x[1]**2
ceq = lambda x : x[1] - ((x[0]-1)**2 + 2)
x_opt = auglag([30.0, 30.0], 100.0, 10.0, 1e-3, f, ceq)
