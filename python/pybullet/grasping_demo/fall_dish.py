import pybullet as p
import time
import pybullet_data
import matplotlib.pyplot as plt
import moviepy.editor as mpy
from base64 import b64encode
from IPython.display import HTML
import numpy as np

def save_video(frames, path):
    clip = mpy.ImageSequenceClip(frames, fps=30)
    clip.write_videofile(path, fps=30)

def play_mp4(path):
    mp4 = open(path, 'rb').read()
    url = "data:video/mp4;base64," + b64encode(mp4).decode()
    return HTML("""<video width=400 controls><source src="%s" type="video/mp4"></video>""" % url)

# 一回だけ実行してください
physicsClient = p.connect(p.GUI)  # ローカルで実行するときは、p.GUI を指定してください
# 床を出現させます
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-10)
timestep = 1. / 240.
p.setTimeStep(timestep)
planeId = p.loadURDF("plane.urdf")
# R2D2を出現させます
cubeStartPos = [0,0,3]  # x,y,z
cubeStartOrientation = p.getQuaternionFromEuler([0,0,3.14])  # roll pitch yaw
boxId = p.loadURDF("dish/plate.urdf",cubeStartPos, cubeStartOrientation)  # humanoid.urdf

frames = []
for t in range (400):
    p.stepSimulation()
    # time.sleep(1./240.)
    if t % 8 == 0:
        width, height, rgbImg, depthImg, segImg = p.getCameraImage(360,240)
        frames.append(rgbImg)

save_video(frames, "sample.mp4")
# 少し高い位置に出現した後、重力に従って落ちて、床との衝突判定を受けて静止します
play_mp4("sample.mp4")
