import numpy as np

def test1():
    a = np.random.randn(3, 3)
    a_inv = np.linalg.inv(a)

    tensor1 = np.array([a] * 5)
    tensor2 = np.array([a_inv] * 5)

    tensor = np.einsum("ijk,ikl->ijl", tensor1, tensor2)

    gtruth = np.array([np.eye(3)] * 5)
    np.testing.assert_almost_equal(tensor, gtruth)

def test2():
    tensor1 = np.array([np.array([1, 2, 3])] * 5)
    a = np.diag([1, 2, 3])
    tensor2 = np.array([a] * 5)
    tensor = np.einsum("ij,ijk->ik", tensor1, tensor2)

    gtruth = np.array([[1, 4, 9]] * 5)
    np.testing.assert_almost_equal(tensor, gtruth)

test1()
test2()
