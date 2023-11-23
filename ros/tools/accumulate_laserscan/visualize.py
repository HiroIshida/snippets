import datetime
import os
import pickle
from skrobot.models.pr2 import PR2
from skrobot.sdf import UnionSDF
from skrobot.model.primitives import PointCloudLink
from skrobot.viewers import TrimeshSceneViewer
from sensor_msgs.msg import PointCloud2, JointState

def find_latest_file(directory):
    files = os.listdir(directory)
    dataset_files = [f for f in files if f.startswith("pointcloud_") and f.endswith(".pkl")]
    if not dataset_files:
        raise FileNotFoundError("No dataset files found in the directory.")
    latest_file = sorted(dataset_files, key=lambda x: datetime.datetime.strptime(x[11:-4], "%Y%m%d-%H%M%S"), reverse=True)[0]
    return os.path.join(directory, latest_file)

def load_latest_dataset(directory):
    latest_file = find_latest_file(directory)
    with open(latest_file, "rb") as f:
        data = pickle.load(f)
    return data

pcloud, jstate = load_latest_dataset("./")

pr2 = PR2()
for joint_name, angle in zip(jstate.name, jstate.position):
    pr2.__dict__[joint_name].joint_angle(angle)

sdf = UnionSDF.from_robot_model(pr2)
pcloud_near = pcloud[sdf(pcloud) < 0.1]
plink = PointCloudLink(pcloud_near)

v = TrimeshSceneViewer()
v.add(plink)
v.add(pr2)
v.show()
import time; time.sleep(1000)
