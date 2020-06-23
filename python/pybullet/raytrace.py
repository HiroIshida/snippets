# https://towardsdatascience.com/simulate-images-for-ml-in-pybullet-the-quick-easy-way-859035b2c9dd
import pybullet as p
import pybullet 
import pybullet_data 
import numpy.random as rn
import numpy as np
import time
import rospkg
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

def get_pose(body):
    return p.getBasePositionAndOrientation(body, physicsClientId=CLIENT)

def get_quat(body):
    return get_pose(body)[1] # [x,y,z,w]

def get_point(body):
    return get_pose(body)[0]

def set_pose(body, pose):
    (point, quat) = pose
    p.resetBasePositionAndOrientation(body, point, quat, physicsClientId=CLIENT)

def set_point(body, point):
    set_pose(body, (point, get_quat(body)))

def raytest_scatter(x_start, b_min, b_max, N):
    w = (b_max - b_min)/N
    pts = []
    mat = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            y = b_min[0] + i * w[0]
            z = b_min[1] + j * w[1]
            pos = np.array([x_start, y, z])

            direction = np.array([ 1.,  0., -0.])
            res = pybullet.rayTest(pos, pos + direction)[0]
            pts.append(np.array(res[3]))
            mat[i, j] = res[2]
    return np.array(pts), mat


rospack = rospkg.RosPack()

CLIENT = p.connect(p.DIRECT)#or pybullet.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

model_urdf = rospack.get_path("eusurdf")
oven = p.loadURDF(model_urdf+"/models/cup/model.urdf")
#hoge = p.loadURDF("plane.urdf")

direction = np.array([1.0, 0.0, 0.0])

# umakuiku
set_point(oven, [0, 0.0, 0.0])
pos = np.array([-0.5 , 0.0  , 0.02])
result1 = p.rayTest(pos, pos + direction * 0.5)[0]

# umakuikanai girigiri
set_point(oven, [0, 0.0, 0.0])
pos = np.array([-0.05 , 0.0  , 0.02])
result2 = p.rayTest(pos, pos + direction * 0.5)[0]

col_pos1 = result1[3]
col_pos2 = result2[3]


def raytest_scatter(x_start, b_min, b_max, N):
    w = (b_max - b_min)/N
    pts = []
    mat = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            y = b_min[0] + i * w[0]
            z = b_min[1] + j * w[1]
            pos = np.array([x_start, y, z])

            direction = np.array([ 1.,  0., -0.])
            res = pybullet.rayTest(pos, pos + direction)[0]
            pts.append(np.array(res[3]))
            mat[i, j] = res[2]
    return np.array(pts), mat

pts, mat = raytest_scatter(-0.5, np.array([-0.2, -0.2]), np.array([0.2, 0.2]), 100)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2]); plt.show()
