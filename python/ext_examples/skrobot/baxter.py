from pathlib import Path
import time
import skrobot
from skrobot.model.robot_model import RobotModel
from skrobot.models.urdf import RobotModelFromURDF
import rospkg

rospack = rospkg.RosPack()
model_urdf_path = Path(rospack.get_path("baxter_description"))
baxter_urdf_path = model_urdf_path / "urdf" / "baxter.urdf"

robot_model = RobotModelFromURDF(urdf_file=str(baxter_urdf_path))
print(robot_model.joint_list)
viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))
viewer.add(robot_model)
viewer.show()
time.sleep(10)
