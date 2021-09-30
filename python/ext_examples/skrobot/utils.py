import numpy as np
def sdf_box(b, c):
    mat = None
    closure_member = {"rotmat": mat}

    def sdf(X):
        rotmat = closure_member["rotmat"]
        if rotmat is not None:
            X = X.dot(closure_member["rotmat"]) # equivalent to inverse (transpose)
        n_pts = X.shape[0]
        dim = X.shape[1]
        center = np.array(c).reshape(1, dim)
        center_copied = np.repeat(center, n_pts, axis=0)
        P = X - center_copied
        Q = np.abs(P) - np.repeat(np.array(b).reshape(1, dim), n_pts, axis=0)
        left__ = np.array([Q, np.zeros((n_pts, dim))])
        left_ = np.max(left__, axis = 0)
        left = np.sqrt(np.sum(left_**2, axis=1))
        right_ = np.max(Q, axis=1)
        right = np.min(np.array([right_, np.zeros(n_pts)]), axis=0)
        return left + right 
    return sdf

