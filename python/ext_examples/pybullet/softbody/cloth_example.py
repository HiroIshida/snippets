import numpy as np
import time
import pybullet as pb
import pybullet_data

def get_pose(body):
    return pb.getBasePositionAndOrientation(body)


def set_point(body, point):
    (point_now, quat_now) = pb.getBasePositionAndOrientation(body)
    pb.resetBasePositionAndOrientation(body, point, quat_now)


pb.connect(pb.GUI)
pb.setPhysicsEngineParameter(numSolverIterations=50)
pb.setTimeStep(timeStep=0.001)
pb.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF
pb.loadURDF("plane.urdf")
#robot_id = pb.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
pb.setGravity(0, 0, -10)

#cloth_scale = 0.10
cloth_scale = 0.30
mass = 0.5
n_cuts = 10
edge_length = (2.0 * cloth_scale) / (n_cuts - 1)
collisionMargin = edge_length / 5.0
base_pos = (0, 0, 3.0)
base_orn = pb.getQuaternionFromEuler([np.pi / 3.0, 0, 0])

cloth_id = pb.loadSoftBody(
        fileName="./bl_cloth_10_cuts.obj",
        basePosition=base_pos,
        baseOrientation=base_orn,
        collisionMargin=collisionMargin,
        scale=cloth_scale,
        mass=mass,
        useNeoHookean=0,
        useBendingSprings=1,
        useMassSpring=1,
        springElasticStiffness=40,
        springDampingStiffness=0.1,
        springDampingAllDirections=0,
        useSelfCollision=1,
        frictionCoeff=1.0,
        useFaceContact=1,)

for _ in range(1000):
    time.sleep(0.01)
    pb.stepSimulation()
    
time.sleep(10)
