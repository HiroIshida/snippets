from dataclasses import dataclass
import osqp
from typing import Optional, Protocol
import numpy as np
import scipy.sparse


class KernelProtocol(Protocol):

    def __call__(self, __X: np.ndarray, __Y: Optional[np.ndarray]) -> np.ndarray:
        ...


class SVM:
    kernel: KernelProtocol
    C: float
    result: Optional["SVM.FitResult"]

    @dataclass
    class FitResult:
        a: np.ndarray
        b: float
        X: np.ndarray
        Y: np.ndarray
        decision_bools: np.ndarray


    def __init__(self, kernel: KernelProtocol, C: float):
        self.kernel = kernel
        self.C = C

    def fit(self, X: np.ndarray, Y: np.ndarray, verbose: bool = False) -> None:
        assert X.dtype == float
        assert Y.dtype == bool
        Y = Y * 2 - 1
        assert X.shape[0] == Y.shape[0]
        assert X.ndim == 2

        # construct P
        gram_matrix = self.kernel(X, None)
        P = gram_matrix * np.outer(Y, Y)
        assert P.shape == (X.shape[0], X.shape[0])
        P_sparse = scipy.sparse.csc_matrix(P)

        # construct q
        q = -np.ones(X.shape[0])

        # construct A
        A = np.zeros((X.shape[0] + 1, X.shape[0]))
        A[0, :] = Y
        A[1:, :] = np.eye(X.shape[0])
        A_sparse = scipy.sparse.csc_matrix(A)

        # construct l and u
        l = np.zeros(X.shape[0] + 1)
        u = np.ones(X.shape[0] + 1) * self.C
        u[0] = 0.0

        prob = osqp.OSQP()
        prob.setup(P_sparse, q, A_sparse, l, u, verbose=verbose, eps_abs=1e-9, eps_rel=1e-9)
        raw_result = prob.solve()

        success_status = osqp.constant("OSQP_SOLVED")
        assert raw_result.info.status_val == success_status
        a = raw_result.x  # lagrange multiplier
        aY_gram_matrix = (a * Y)[:, np.newaxis] * gram_matrix
        b = (np.sum(Y) - np.sum(aY_gram_matrix)) / len(Y)

        # find support vectors
        decision_values = np.sum(aY_gram_matrix, axis=0) + b
        print(decision_values)
        decision_bools = np.abs((np.abs(decision_values) - 1.0)) < 1e-1
        print(decision_bools)
        self.result = SVM.FitResult(a, b, X, Y, decision_bools)

    def decision_function(self, X: np.ndarray, use_sparse: bool = False) -> np.ndarray:
        assert X.dtype == float
        assert X.ndim == 2
        assert self.result is not None

        if use_sparse:
            X_data = self.result.X[self.result.decision_bools, :]
            aY = (self.result.a * self.result.Y)[self.result.decision_bools]
        else:
            X_data = self.result.X
            aY = self.result.a * self.result.Y

        Y = np.sum(aY[:, np.newaxis] * self.kernel(X_data, X), axis=0) + self.result.b
        print(Y)
        return Y

    def predict(self, X: np.ndarray, use_sparse: bool = False) -> np.ndarray:
        assert X.dtype == float
        assert X.ndim == 2
        return self.decision_function(X, use_sparse=use_sparse) > 0.0


if __name__ == "__main__":
    from sklearn.metrics.pairwise import rbf_kernel
    X = np.random.randn(100, 2)
    # inside circle or not
    Y = np.linalg.norm(X, axis=1) < 1.0
    svm = SVM(rbf_kernel, 10.0)
    svm.fit(X, Y, verbose=True)
    # print(np.abs(svm.decision_function(X)) < 1.0)
    # Y_predict = svm.predict(X, True)
    # print(np.sum(Y == Y_predict) / len(Y))


    import matplotlib.pyplot as plt
    x_mesh = np.linspace(-1.5, 1.5, 30)
    y_mesh = x_mesh
    mesh_grid = np.meshgrid(x_mesh, y_mesh)
    pts = np.array(list(zip(mesh_grid[0].flatten(), mesh_grid[1].flatten())))
    preds_ = svm.decision_function(pts)
    preds = preds_.reshape(30, 30)

    # compute accuracy
    Y_predict = svm.predict(X, False)
    print(np.sum(Y == Y_predict) / len(Y))

    fig, ax = plt.subplots()
    cs = ax.contour(x_mesh, y_mesh, preds, levels = [-1.0, 0.0, 1.0], cmap = 'jet', zorder=1)
    ax.scatter(X[:, 0], X[:, 1])
    plt.show()
