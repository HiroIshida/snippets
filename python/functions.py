import numpy as np

def modified_banana_function(x_):
    a = 1.0
    b = 10.0
    x = ((x_ - np.array([0, -0.5])) * 4)
    if x[0] > 0:
        x[0] = -x[0] 

    binary = (a - x[0])**2 + b*(x[1]-x[0]**2)**2 - 10 < 0
    return binary
