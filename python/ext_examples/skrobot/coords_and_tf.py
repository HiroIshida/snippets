import skrobot
from skrobot.coordinates import Coordinates
from skrobot.coordinates import make_coords, rpy_matrix
from skrobot.coordinates.math import matrix2quaternion, wxyz2xyzw

def tf_from_xytheta(xytheta):
    x, y, theta = xytheta
    pos = [x, y, theta]
    rot = rpy_matrix(*[theta, 0, 0])
    quat = wxyz2xyzw(matrix2quaternion(rot))
    tf = [pos, quat.tolist()]
    return tf

if __name__=='__main__':
    import time
    xytheta = [0.1, 0.1, 0.3]
    ts = time.time()
    tf = tf_from_xytheta(xytheta)
    print(time.time() - ts)
