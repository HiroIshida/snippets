import pybullet as p
import pybullet
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

planeId = p.loadURDF("plane.urdf")
for _ in range(10):
    vis_id = pybullet.createCollisionShape(pybullet.GEOM_BOX, halfExtents=(0.2, 0.2, 0.2))
    body_id = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=vis_id)
    set_point(body_id, [rn.randn()*0.3, rn.randn()*0.3, 1.2 + rn.random()*2])

time.sleep(2.0)
for i in range(1000):
    if i%5==1:
        time.sleep(0.1)
    p.stepSimulation()
