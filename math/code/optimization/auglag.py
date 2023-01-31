import copy
import time
import numpy as np 
import scipy.optimize as opt

# see Nocedal & wright, Numerical Optimization, p 516

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

def check_kkt_condition(x_sol, lam_sol, f, ceq, cineq): 
    Lag = lambda x: f(x)[0] - lam_sol[0] * ceq(x)[0] - lam_sol[1] * cineq(x)[0]
    grad_Lag = compute_grad(Lag, x_sol)

    eps = 1e-3
    assert np.linalg.norm(grad_Lag) < eps, "grad of lag is {0}".format(grad_Lag)
    print("passed condition of laggrad")
    assert ceq(x_sol)[0] < eps
    print("passed eq feasiblity")

    assert cineq(x_sol) > 0 - eps
    print("passed ineq feasiblity")

    assert lam_sol[1] > 0 - eps
    print("passed multiplier condition")

    assert abs(lam_sol[0] *  ceq(x_sol)[0]) < eps
    assert abs(lam_sol[1] *  cineq(x_sol)[0]) < eps
    print("passed dual condition")


def auglag(x0, lam0, mu, tau, f, ceq, cineq):
    x_now = np.array(x0)
    lam_now = np.array(lam0)
    mu_now = mu
    tau_now = tau

    def psi(t, sigma, mu):
        if t - mu * sigma < 0.0:
            return - sigma * t + 1/(2*mu) * t**2
        else:
            return - 0.5 * mu * sigma ** 2

    for i in range(50):
        def Lag(x):
            obj_val, obj_grad = obj(x)
            ceq_val, ceq_grad = ceq(x)
            cineq_val, cineq_grad = cineq(x)

            lag_val =  obj_val - lam_now[0] * ceq_val + 1.0/(2 * mu_now) * ceq_val ** 2 \
                    + psi(cineq_val, lam_now[1], mu_now)
            return lag_val

        #res = opt.minimize(Lag, x_now, method='BFGS', options={'disp': False, 'gtol': tau})

        ts = time.time()
        res = opt.least_squares(Lag, x_now, ftol=1e-03)
        te = time.time()
        print(te - ts)

        x_star_approx = res.x

        x_now = x_star_approx
        lam_now[0] -= ceq(x_star_approx)[0]/mu_now
        lam_now[1] = max(lam_now[1] - cineq(x_star_approx)[0]/mu_now, 0)
        mu_now *= 0.9 # if I set it to smaller, the x_opt diverges ...
        tau_now *= 0.1

    check_kkt_condition(x_now, lam_now, f, ceq, cineq)
    return x_now

def obj(x): 
    f = x[0] ** 2 + x[1]**2
    grad = 2 * x
    return f, grad

def ceq(x):
    f = x[1] - ((x[0]-1)**2 + 1)
    grad = np.array([- 2 * x[0], 1])
    return f, grad

def cineq(x):
    f = x[1] - (x[0] + 2)
    grad = np.array([-1, 1])
    return f, grad

x_opt = auglag([30.0, 30.0], [100.0, 100.0], 10.0, 1e-3, obj, ceq, cineq)
