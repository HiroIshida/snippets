from math import *
import numpy as np
import sympy
from sympy import Matrix
from sympy.vector.orienters import AxisOrienter
from skrobot.coordinates.math import rotation_matrix


def expr_rotation_x(theta):
    return sympy.Matrix([
        [1, 0, 0],
        [0, sympy.cos(theta), -sympy.sin(theta)],
        [0, sympy.sin(theta), sympy.cos(theta)]
    ])


def expr_rotation_y(theta):
    return sympy.Matrix([
        [sympy.cos(theta), 0, sympy.sin(theta)],
        [0, 1, 0],
        [-sympy.sin(theta), 0, sympy.cos(theta)]
    ])


def expr_rotation_z(theta):
    return sympy.Matrix([
        [sympy.cos(theta), -sympy.sin(theta), 0],
        [sympy.sin(theta), sympy.cos(theta), 0],
        [0, 0, 1]
    ])


def expr_rotation_matrix(roll, pitch, yaw):
    return expr_rotation_z(yaw) * expr_rotation_y(pitch) * expr_rotation_x(roll)


arg_names = ("roll", "pitch", "yaw")
arg_exprs = sympy.symbols(arg_names)
rpy_mat_expr = expr_rotation_matrix(*arg_exprs)
x_axis_expr = rpy_mat_expr[:, 0]
jac_expr = x_axis_expr.jacobian(arg_exprs)
print(jac_expr)

f_jac_axis = sympy.lambdify(arg_exprs, jac_expr, modules="numpy")
jac = f_jac_axis(0.2, 0.2, 0.2)
print(jac)

def compute_by_hand(roll, pitch, yaw):
    jac =  np.array([
        [0, -sin(pitch)*cos(yaw), -sin(yaw)*cos(pitch)],
        [0, -sin(pitch)*sin(yaw),  cos(pitch)*cos(yaw)],
        [0,          -cos(pitch),                    0]])
    return jac


# check computation time
import time
ts = time.time()
for _ in range(10000):
    compute_by_hand(0.2, 0.2, 0.2)
print("compute by hand: {} sec".format(time.time() - ts))

ts = time.time()
for _ in range(10000):
    f_jac_axis(0.2, 0.2, 0.2)
print("compute by sympy: {} sec".format(time.time() - ts))
