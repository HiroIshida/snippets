import numpy as np
import pickle
import scipy.interpolate  

def quaternion_matrix(quaternion): # fetched from tf.transformations (ver 2009)
    _EPS = np.finfo(float).eps * 4.0
    q = np.array(quaternion[:4], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.identity(4)
    q *= np.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=np.float64)

class CostmapFunctionData:
    def __init__(self, msg, tf_base_to_odom):
        self.msg = msg
        self.tf_base_to_odom = tf_base_to_odom

    def convert2sdf(self):
        info = self.msg.info
        n_grid = np.array([info.width, info.height])
        res = info.resolution
        origin = info.origin

        b_min = np.zeros(2)
        b_max = n_grid * res
        xlin, ylin = [np.linspace(b_min[i], b_max[i], n_grid[i]) for i in range(2)]

        tmp = np.array(self.msg.data).reshape((n_grid[1], n_grid[0])) # about to be transposed!!
        arr = tmp.T # [IMPORTANT] !!
        fp_wrt_map = scipy.interpolate.RegularGridInterpolator((xlin, ylin), arr, 
                method='linear', bounds_error=False, fill_value=0.0) 

        def base_to_map(P): 
            n_points = len(P)
            pos_base_to_odom, rot_base_to_odom = self.tf_base_to_odom
            M = quaternion_matrix(rot_base_to_odom)[:2, :2]
            points = P.dot(M.T) + np.repeat(
                    np.array([[pos_base_to_odom[0], pos_base_to_odom[1]]]), n_points, 0)
            mappos_origin = np.array([[origin.position.x, origin.position.y]])
            points = points - np.repeat(mappos_origin, n_points, 0)
            return points

        fp_wrt_base = lambda X: fp_wrt_map(base_to_map(X))
        return fp_wrt_base

    def save(self, filename="costmapf.pickle"):
        with open(filename, "wb") as f:
            pickle.dump(self, f)
