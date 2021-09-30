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

CLIENT = p.connect(p.GUI)#or pybullet.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
p.setGravity(0,0,-10)

tray_id = p.loadURDF("./tray/tray.urdf")
ball_ids = [p.loadURDF("sphere_small.urdf") for i in range(200)]
for ball_id in ball_ids:
    set_point(ball_id, [rn.randn()*0.05, rn.randn()*0.05, 0.2 + rn.random()*2])

planeId = p.loadURDF("plane.urdf")
time.sleep(2.0)
for i in range(1000):
    if i%5==1:
        time.sleep(0.1)
    p.stepSimulation()
