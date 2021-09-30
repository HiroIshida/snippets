import pybullet as p
import pybullet_data
import utils

useMaximalCoordinates = False

p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())

pole = p.loadURDF("cartpole.urdf", [0, 0, 0], useMaximalCoordinates=useMaximalCoordinates)


for i in range(2):
  p.setJointMotorControl2(pole, i, p.POSITION_CONTROL, targetPosition=0, force=500)

for i in range(2):
    p.resetJointState(pole, i, 0.3, targetVelocity=0.3)

timeStep = 0.0000001
p.setTimeStep(timeStep)
while p.isConnected():
    position, velocity, _, _ = p.getJointState(pole, 0)
    print(velocity)
    p.stepSimulation()
