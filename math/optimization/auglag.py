import numpy as np 
import scipy.optimize as opt

def auglag(x0, lam0, mu, tau, f, ceq):
    x_now = x0
    lam_now = lam0
    mu_now = mu
    tau_now = tau

    for i in range(20):
        Lag = lambda x: f(x) - lam_now * ceq(x) + 1.0/(2 * mu_now) * ceq(x)**2
        #res = opt.minimize(Lag, x_now, method='BFGS', options={'disp': True, 'gtol': tau})
        res = opt.minimize(Lag, x_now, method='BFGS', options={'disp': False, 'gtol': tau})
        x_star_approx = res.x

        x_now = x_star_approx
        lam_now -= ceq(x_star_approx)/mu
        mu *= 0.8
        tau_now *= 1.0
        print(x_now)

f = lambda x : x[0] ** 2 + x[1]**2
ceq = lambda x : x[1] - (x[0]**2 + 2)
auglag([30.0, 30.0], 100.0, 10.0, 1e-3, f, ceq)
