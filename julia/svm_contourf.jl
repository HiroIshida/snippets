import LIBSVM
import PyPlot

function simple_data()
    x1 = [0, 0.2, 0.5, 0,   -0.2, -0.3, 0.8, 0.8, 0.5, 0.4]
    x2 = [0, 0.9, 0.1, 0.9, -0.9, 0.1, -0.2, 0.2, 0.5, -0.3]
    X = transpose(hcat(x1, x2))
    y = [true, true, true, false, false, false, false, false, false, false]
    return X, y
end

X_sample, Y_sample = simple_data()
model = LIBSVM.svmtrain(X_sample, Y_sample; verbose=false, gamma = 10.0)

xlin = range(0, 1, length=100)
ylin = range(0, 1, length=100)
f(x) = LIBSVM.svmpredict(model, reshape(x, 2, 1))[2][1]
fs = [f([x y]) for x in xlin, y in ylin]
PyPlot.contourf(xlin, ylin, fs')
PyPlot.plt.colorbar()
