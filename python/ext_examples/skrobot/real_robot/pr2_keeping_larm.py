import numpy as np
import time
import tqdm
from skrobot.interfaces import PR2ROSRobotInterface
from skrobot.models.pr2 import PR2
from skrobot.viewers import PyrenderViewer
from plainmp.robot_spec import PR2LarmSpec
from plainmp.ik import solve_ik
from plainmp.utils import set_robot_state

# this script is for testing pr2'real robot's accuracy of FK
# because all the configurations in q_list_sorted are generated by IK
# for the same gripper pose, the robot's end effector should be at the same position
# However, unfortunately, the real robot's end effector is not at the same position
# more specifically, typically has 1cm or more of variation

pr2 = PR2(use_tight_joint_limit=False)
pr2.reset_manip_pose()
spec = PR2LarmSpec()
spec.reflect_skrobot_model_to_kin(pr2)
cst = spec.create_gripper_pose_const([0.7, 0.3, 0.9, 0, 0, 0])
lb, ub = spec.angle_bounds()

q_list = []
for _ in tqdm.tqdm(range(300)):
    while True:
        ret = solve_ik(cst, None, lb + 0.2, ub - 0.2)
        if ret.success:
            break
    q_list.append(ret.q)

# q_list must lie on 1 dimensional manifold thus
q_list_sorted = [q_list.pop()]
while len(q_list) > 0:
    arr = np.array(q_list)
    query = q_list_sorted[-1]
    dists_from_query = np.linalg.norm(arr - query, axis=1)
    min_idx = np.argmin(dists_from_query)
    q_list_sorted.append(q_list.pop(min_idx))

simulation = False
if simulation:
    viewer = PyrenderViewer()
    viewer.add(pr2)
    viewer.show()

    for q in tqdm.tqdm(q_list_sorted):
        set_robot_state(pr2, spec.control_joint_names, q)
        viewer.redraw()
        time.sleep(0.04)
else:
    ri = PR2ROSRobotInterface(pr2)
    init_flag = True
    for q in tqdm.tqdm(q_list_sorted):
        set_robot_state(pr2, spec.control_joint_names, q)
        ri.angle_vector(pr2.angle_vector(), time=1)
        if init_flag:
            ri.wait_interpolation()
        else:
            time.sleep(0.7)
