import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
import numpy as np
from bayes_opt import BayesianOptimization, UtilityFunction


assert False, "please use botorch version"
assert False, "please use botorch version"
assert False, "please use botorch version"
assert False, "please use botorch version"
assert False, "please use botorch version"
assert False, "please use botorch version"
assert False, "please use botorch version"
assert False, "please use botorch version"
assert False, "please use botorch version"
assert False, "please use botorch version"
assert False, "please use botorch version"


def f_bench(x_dict):

    if isinstance(x_dict, Dict):
        x = np.array(list(x_dict.values()))
    else:
        x = x_dict
    t1 = np.sum(x ** 4)
    t2 = - 16 * np.sum(x ** 2)
    t3 = 5 * np.sum(x)
    return -0.5 * (t1 + t2 + t3)


D = 100
d = 3
pbounds = {"x{}".format(i): (-5, 5) for i in range(D)}
x_optimal = -np.ones(D) * 2.903534
f_optimal = f_bench(x_optimal)

use_embedding = False

if use_embedding:
    M = np.random.randn(D, d)
    pbounds_embed = {"y{}".format(i): (-5 * np.sqrt(d), 5 * np.sqrt(d)) for i in range(d)}
    # pbounds_embed = {"y{}".format(i): (-20 * np.sqrt(d), 20 * np.sqrt(d)) for i in range(d)}

    def f_bench_embed(z_dict):
        z = np.array(list(z_dict.values()))
        x = M.dot(z_arr).flatten()
        return f_bench(x)

    optimizer = BayesianOptimization(
        f=f_bench_embed,
        pbounds=pbounds_embed,
        random_state=1,
    )
else:
    optimizer = BayesianOptimization(
        f=f_bench,
        pbounds=pbounds,
        random_state=1,
    )


utility = UtilityFunction()

y_list = []
for i in range(150):
    if i > 0:
        pass

    try:
        if use_embedding:
            z = optimizer.suggest(utility)
            z_arr = np.expand_dims(np.array(list(z.values())), axis=1)
            # x = M.dot(np.expand_dims(z_arr, axis=1)).flatten()
            x = M.dot(z_arr).flatten()
        else:
            x = optimizer.suggest(utility)
        y = f_bench(x)
        if use_embedding:
            optimizer.register(params=z, target=y)
        else:
            optimizer.register(params=x, target=y)
        y_list.append(y)
        gap = f_optimal - y
        print("{}=> y: {}, gap: {}".format(i, y, gap))
    except AttributeError as e:
        print(e)
        print("error. skipped")
        pass
plt.plot(y_list)
plt.show()
