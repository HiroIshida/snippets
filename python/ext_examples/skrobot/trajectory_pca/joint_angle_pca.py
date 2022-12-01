import numpy as np
import time
from skrobot.viewers import TrimeshSceneViewer
from skrobot.models import PR2
from sklearn.decomposition import PCA

joint_names = [
    "r_shoulder_pan_joint",
    "r_shoulder_lift_joint",
    "r_upper_arm_roll_joint",
    "r_elbow_flex_joint",
    "r_forearm_roll_joint",
    "r_wrist_flex_joint",
    "r_wrist_roll_joint",
]


def get_q():
    angles = []
    for jn in joint_names:
        joint = robot.__dict__[jn]
        angles.append(joint.joint_angle())
    return np.array(angles)

def set_q(vec):
    angles = []
    for jn, angle in zip(joint_names, vec):
        joint = robot.__dict__[jn]
        joint.joint_angle(angle)


robot = PR2(use_tight_joint_limit=False)

robot.init_pose()
q1 = get_q()
robot.reset_manip_pose()
q2 = get_q()

Q = np.array([q1, q2])

pca = PCA(n_components=1)
pca.fit(Q)
Z = pca.transform(Q)

n = 20
step = (Z[1] - Z[0]) / n

q_list = []
for i in range(n):
    z = Z[0] + step * i
    q_sampled = pca.inverse_transform(np.expand_dims(z, axis=0))[0]
    q_list.append(q_sampled)

viewer = TrimeshSceneViewer()
viewer.add(robot)
viewer.show()

for q in q_list:
    set_q(q)
    viewer.redraw()
    time.sleep(0.3)

print('==> Press [q] to close window')
while not viewer.has_exit:
    time.sleep(0.1)
    viewer.redraw()
