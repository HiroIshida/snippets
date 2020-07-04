import numpy as np
# http://hooktail.sub.jp/mechanics/inertiaTable1/

def sphere(M, r):
    I = 2/5*M * r**2
    return [I, I, I]

# point symetric around z axis
def cylinder(M, r, h):
    Ix = ((r**2/2.0) + (h**2/12.0)) * M
    Iz = M * r**2
    return [Ix, Ix, Iz]

def box(M, a, b, c):
    Ix = 1/3.0 * (b**2 + c**2) * M
    Iy = 1/3.0 * (c**2 + a**2) * M
    Iz = 1/3.0 * (a**2 + b**2) * M
    return [Ix, Iy, Iz]
