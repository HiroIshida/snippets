import pybullet as p
import time
import math
import pybullet_data

useGui = True

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
#p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)

#p.loadURDF("samurai.urdf")
p.loadURDF("r2d2.urdf", [3, 3, 1])

rayFrom = []
rayTo = []
rayIds = []

numRays = 1024

rayLen = 13

rayHitColor = [1, 0, 0]
rayMissColor = [0, 1, 0]

replaceLines = True

for i in range(numRays):
  rayFrom.append([0, 0, 1])
  rayTo.append([
      rayLen * math.sin(2. * math.pi * float(i) / numRays),
      rayLen * math.cos(2. * math.pi * float(i) / numRays), 1
  ])

infos = p.rayTestBatch(rayFrom, rayTo) 
list(filter(lambda info: info[0] != -1, infos))
