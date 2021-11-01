import os
import pybullet as pb
import pybullet_data 
import matplotlib.pyplot as plt

from mimic.dataset import KinematicsDataset
from mimic.models import KinemaNet
from mimic.trainer import TrainCache

pbdata_path = pybullet_data.getDataPath()
urdf_path = os.path.join(pbdata_path, 'kuka_iiwa', 'model.urdf')
joint_names = ['lbr_iiwa_joint_{}'.format(idx+1) for idx in range(7)]
link_names = ['lbr_iiwa_link_7']

dataset = KinematicsDataset.from_urdf(urdf_path, joint_names, link_names, n_sample=10)
tcache = TrainCache[KinemaNet].load('kinematics', KinemaNet)
tcache.visualize()
plt.show()
model = tcache.best_model

inp, out = dataset[1]
out_exp = model.layer(inp)
print(out)
print(out_exp)

