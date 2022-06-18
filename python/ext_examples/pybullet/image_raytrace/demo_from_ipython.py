# https://github.com/irvingvasquez/nbv_regression/blob/main/range_simulation_tutorial/pybullet_sim_tutorial.ipynb
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import math

import pybullet as pb 
import pybullet_data

from pybullet_object_models import ycb_objects

# Start the pybullet server
physicsClient = pb.connect(pb.GUI)
pb.resetDebugVisualizerCamera(3, 90, -30, [0.0, -0.0, -0.0])
pb.setTimeStep(1 / 240.)

# Insert a plane in the environment
pb.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = pb.loadURDF('plane.urdf')

# Insert some objects
flags = pb.URDF_USE_INERTIA_FROM_FILE

n_objects = 30
offset = 0.2
for i in range(n_objects):
    z_pos = 0 + i*offset
    obj_id = pb.loadURDF(os.path.join(ycb_objects.getDataPath(), 'YcbBanana', "model.urdf"), [0, 0.0, z_pos], flags=flags)


pb.setGravity(0, 0, -9.8)

# Start simulation
import time
time.sleep(0.1)
for i in range(1000):
    if i%20==1:
        time.sleep(0.01)
    pb.stepSimulation()

# Define a camera view matrix
camPos = [0, -2, 0.5]
camTarget = [0, 0, 0]
camUp = [0, 1, 0]
viewMatrix = pb.computeViewMatrix(
    cameraEyePosition = camPos,
    cameraTargetPosition = camTarget,
    cameraUpVector=camUp)

# Define projection matrix
nearVal = 0.01
farVal = 5.1

projectionMatrix = pb.computeProjectionMatrixFOV(
    fov=30.0,
    aspect=1.0,
    nearVal=nearVal,
    farVal=farVal)

# Get an image
imgW, imgH, rgbImg, depthImg, segImg = pb.getCameraImage(
    width=128, 
    height=128,
    viewMatrix=viewMatrix,
    projectionMatrix=projectionMatrix,
    renderer=pb.ER_BULLET_HARDWARE_OPENGL
)

stepX = 1
stepY = 1        
pointCloud = np.empty([np.int32(imgH/stepY), np.int32(imgW/stepX), 4])

projectionMatrix = np.asarray(projectionMatrix).reshape([4,4],order='F')

viewMatrix = np.asarray(viewMatrix).reshape([4,4],order='F')

tran_pix_world = np.linalg.inv(np.matmul(projectionMatrix, viewMatrix))

for h in range(0, imgH, stepY):
    for w in range(0, imgW, stepX):
            x = (2*w - imgW)/imgW
            y = -(2*h - imgH)/imgH  # be careful！ deepth and its corresponding position
            # duda
            z = 2*depthImg[h,w] - 1
            #z = realDepthImg[h,w]
            pixPos = np.asarray([x, y, z, 1])
            #print(pixPos)
            position = np.matmul(tran_pix_world, pixPos)
            pointCloud[np.int32(h/stepY), np.int32(w/stepX), :] = position / position[3]
            
print(pointCloud.shape)

# Creating 3D figure
xs = []
ys = []
zs = []

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

ax = plt.axes(projection='3d')

for h in range(0, imgH, stepY):
    for w in range(0, imgW, stepX):
        [x,y,z,_] = pointCloud[h,w]
        xs.append(x)
        ys.append(y)
        zs.append(z)
        

n = 10000
ax.scatter(xs[:n], ys[:n], zs[:n], c=zs[:n], marker='.');
ax.view_init(75, 45)
ax.set_xlabel("x-axis")
ax.set_ylabel("y-axis")
ax.set_zlabel("z-axis")
plt.draw()
plt.show()
