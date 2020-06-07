# as for pybullet robot see: https://alexanderfabisch.github.io/pybullet.html
import rospkg
import pybullet 
import pybullet_data

rospack = rospkg.RosPack()

models_dir = rospack.get_path("eusurdf")

physicsClient = pybullet.connect(pybullet.GUI)#or p.DIRECT for non-graphical version
#physicsClient = p.DIRECT
pybullet.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF
pybullet.setGravity(0,0,-10)

#planeId = pybullet.loadURDF("plane.urdf")

#robot = pybullet.loadURDF("./pr2_description/pr2.urdf")

robot = pybullet.loadURDF("/opt/ros/kinetic/share/fetch_description/robots/fetch.urdf")

table_file = models_dir + "/models/room73b2-karimoku-table/model.urdf"
table = pybullet.loadURDF(table_file, basePosition = [1, 0.0, 0.0])

#Qoven_file = models_dir + "/models/toshiba-microwave-oven/model.urdf"
#oven = pybullet.loadURDF(oven_file, basePosition = [1, 0.0, 0.7])
