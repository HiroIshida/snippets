import numpy as np
import numpy.random as rn

B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
B_ = B.reshape(4, 1, 3)

C = np.vstack((rn.randn(3, 7), rn.randn(3, 7), rn.randn(3, 7), rn.randn(3, 7)))
C_ = C.reshape(4, 3, 7)
np.matmul(B_, C_)

