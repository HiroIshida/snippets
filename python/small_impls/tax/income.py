import numpy as np
import matplotlib.pyplot as plt

def specific(x):
    if x < 195:
        return (0.05, 0)
    elif x < 330: 
        return (0.1, 9.75)
    elif x < 694:
        return (0.2, 42.75)
    elif x < 899:
        return (0.23, 63.6)
    elif x < 1800:
        return (0.33, 153.6)
    elif x < 4000:
        return (0.4, 279.6)
    else:
        return (0.45, 479.6)

def calc(x):
    rate, kojo = specific(x)
    return x * rate - kojo

X = np.linspace(0, 1000)
Y = np.array([calc(x) for x in X])

plt.plot(X, Y)
plt.show()

