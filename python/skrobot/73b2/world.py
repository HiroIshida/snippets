# https://gist.github.com/furushchev/f55fb768af9d73ad903fe1cdd94bb8b4
import skrobot
from skrobot.models.primitives import MeshLink
from skrobot.models.urdf import RobotModelFromURDF
import os.path as osp

robot_model = RobotModelFromURDF(urdf_file=osp.abspath("./models/room73b2-hitachi-fiesta-refrigerator_fixed/model.urdf"))

viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))
viewer.add(robot_model)
viewer.show()
#robot_model = RobotModelFromURDF(urdf_file=osp.abspath("fetch_description/fetch.urdf"))

"""
<xacro:include filename="$(find eusurdf)/models/room73b2-hitachi-fiesta-refrigerator_fixed/model.urdf.xacro"/>
    <room73b2-hitachi-fiesta-refrigerator_fixed name="room73b2-hitachi-fiesta-refrigerator_0" parent="room73b2_root_link">
      <origin xyz="5.72 1.48 0.0" rpy="0.0 0.0 3.14159"/>
    </room73b2-hitachi-fiesta-refrigerator_fixed>

"""
