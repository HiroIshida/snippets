# these values are determined using 
# https://github.com/DKenefake/OptimalPoly

import numpy as np

def sin_approx_remez6(x):  # 6th order in remez but actually 5th order
  # -0.5 * np.pi <= x <= 0.5 * np.pi
  # [-2.4196883254554248e-09, 0.9996967783286441, 2.1180127879771746e-08, -0.16567308035618675, -1.9539509102297506e-08, 0.007514376473825604, 4.6011871493990204e-09]
  xx = x * x
  xxx = xx * x
  xxxxx = xxx * xx
  return 0.9996967783286441 * x - 0.16567308035618675 * xxx + 0.007514376473825604 * xxxxx


def sin_approx_remez6_full_range(x):
  # to [-np.pi, np.pi]
  x = np.remainder(x + np.pi, 2 * np.pi) - np.pi
  x += (x < -0.5 * np.pi) * ( -np.pi - 2 * x) 
  x += (x > 0.5 * np.pi) * (np.pi - 2 * x)
  return sin_approx_remez6(x)


lb = - 4 * np.pi
ub = + 4 * np.pi
x = np.linspace(lb, ub, 1000)
y = np.sin(x)
y_approx = sin_approx_remez6_full_range(x)

import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 1)
axes[0].plot(x, y, label='sin(x)')
axes[0].plot(x, y_approx, label='sin_approx_remez6(x)')
axes[0].legend()
axes[1].plot(x, y - y_approx, label='sin(x) - sin_approx_remez6(x)')
axes[1].legend()
plt.show()
