from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np
dataset1 = [
        [[0, 0], [0, 0], [1, 1]],
        [0, 0, 1]
        ]
dataset2 = [
        [[0, 0], [1, 1]],
        [0, 1]
        ]
clf = svm.SVC()
myfit = lambda dataset : clf.fit(dataset[0], dataset[1])
myfit(dataset2)

X = np.linspace(-0.5, 1.5, 30)
Y = X
mesh_grid = np.meshgrid(X, Y)
pts = np.array(zip(mesh_grid[0].flatten(), mesh_grid[1].flatten()))
preds_ = clf.predict(pts)
preds = preds_.reshape(30, 30)

fig, ax = plt.subplots()
cs = ax.contour(X, Y, preds, levels = [0.0], cmap = 'jet', zorder=1)
plt.show()


