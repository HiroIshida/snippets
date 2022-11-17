import numpy as np
import matplotlib.pyplot as plt
import pickle
with open("result.pkl", "rb") as f:
    result = pickle.load(f)

fig, ax = plt.subplots()

for matrix_size in result.keys():
    subresult = result[matrix_size]
    n_process_list = list(subresult.keys())
    elapsed_time_list = np.array(list(subresult.values()))
    speedups = elapsed_time_list[0] / elapsed_time_list
    ax.plot(n_process_list, speedups, label=matrix_size)

ax.legend(loc="upper left", borderaxespad=0, fontsize=10, framealpha=1.0)
plt.show()
