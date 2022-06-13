
import pybullet as p
import pybullet_data 
import numpy.random as rn
import time

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

CLIENT = p.connect(p.DIRECT)#or pybullet.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
p.setGravity(0,0,-10)

tray_id = p.loadURDF("./tray/tray.urdf")
id1 = p.loadURDF("sphere_small.urdf")
id2 = p.loadURDF("sphere_small.urdf")

set_point(id1, [0., 0., 0.])
set_point(id2, [1.0, 0., 0.])
ret = p.getClosestPoints(id1, id2, distance=2.0)  
print(len(ret))  # 1

ret = p.getClosestPoints(id1, id2, distance=0.8)  
print(len(ret))  # 0:w

