using Plots
X = rand(2, 100)
scatter(X[1, :], X[2, :], seriestype=:scatter, color = "red", markersize = 2)

## 3d guriguri 
using Plots
Plots.plotly()
plotly()
X = randn(3, 1000)
plt = scatter(X[1, :], X[2, :], X[3, :])
