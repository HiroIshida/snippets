from sympy import *
from sympy.diffgeom.rn import R2, R2_r 
from sympy.diffgeom import BaseCovarDerivativeOp
from sympy.diffgeom import metric_to_Christoffel_2nd, TensorProduct, metric_to_Christoffel_1st
TP = TensorProduct
R = symbols("R")
theta, r, dtheta, dr = symbols("theta, r, dtheta, dr")
metric = TP(R2.dtheta, R2.dtheta) * R**2 + TP(R2.dr, R2.dr) * R**2 * sin(R2.theta) **2
#ch = metric_to_Christoffel_2nd(metric)
ch = metric_to_Christoffel_2nd(metric)

chris = lambda i, j, k: ch[i, j, k]

def curv_expr(ch, i, j, k, l):
    term1 = diff(chris(i, k, l), theta)
    term2 = diff(chris(j, k, l), r)

    term3 = sum([chris(i, k, m) * chris(j, m, l) for m in range(2)])
    term4 = sum([chris(j, k, m) * chris(i, m, l) for m in range(2)])

    whole = term1 - term2 + term3 - term4
    return whole

for i in range(2):
    for j in range(2):
        for k in range(2):
            for l in range(2):
                print(curv_expr(ch, i, j, k, l))


