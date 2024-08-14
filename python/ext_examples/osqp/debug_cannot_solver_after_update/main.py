import numpy as np
import osqp
import pickle
from scipy import sparse

# https://groups.google.com/g/osqp/c/ZFvblAQdUxQ
# https://groups.google.com/g/osqp/c/ZFvblAQdUxQ

if __name__ == "__main__":
    update = True

    with open("/tmp/osqp1.pkl", "rb") as f:
        data = pickle.load(f)
        P1, A1, l1, u1 = data["P"], data["A"], data["l"], data["u"]

    prob = osqp.OSQP()
    prob.setup(P=P1, q=None, l=l1, u=u1, A=A1)
    ret = prob.solve()
    assert ret.info.status_val == osqp.constant("OSQP_SOLVED")

    with open("/tmp/osqp2.pkl", "rb") as f:
        data = pickle.load(f)
        A2, l2, u2 = data["A"], data["l"], data["u"]

    # check that A2 has same sparsity pattern as A1
    assert np.all(A2.nonzero()[0] == A1.nonzero()[0])
    assert np.all(A2.nonzero()[1] == A1.nonzero()[1])

    if update:
        # with update = True, the problem cannot be solved
        if not A2.has_sorted_indices:
            A2.sort_indices()
        prob.update(Ax=A2.data, l=l2, u=u2)
        ret = prob.solve()
        assert ret.info.status_val == osqp.constant("OSQP_SOLVED")
    else:
        # but if we setup() the problem again, it works
        prob = osqp.OSQP()
        prob.setup(P=P1, q=None, l=l2, u=u2, A=A2)
        ret = prob.solve()
        assert ret.info.status_val == osqp.constant("OSQP_SOLVED")
