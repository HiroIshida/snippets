#!/usr/bin/env python
import time

import numpy as np

import skrobot
from skrobot.model.primitives import Axis
from skrobot.model.primitives import Box
from skrobot.planner import tinyfk_sqp_plan_trajectory
from skrobot.planner import TinyfkSweptSphereSdfCollisionChecker
from skrobot.planner import ConstraintManager
from skrobot.planner import ConstraintViewer
from skrobot.planner.utils import get_robot_config
from skrobot.planner.utils import set_robot_config
from skrobot.planner.utils import update_fksolver
from pr2opt_common import *

# initialization stuff
np.random.seed(0)
robot_model = pr2_init()

fridge = Box(extents=[0.8, 0.8, 2.0], with_sdf=True)
fridge.translate([2.2, 2.0, 1.0])

sscc = TinyfkSweptSphereSdfCollisionChecker(lambda X: fridge.sdf(X), robot_model)
for link in rarm_coll_link_list(robot_model):
    sscc.add_collision_link(link)
joint_list = rarm_joint_list(robot_model)

with_base = True

# constraint manager
n_wp = 14
fksolver = sscc.fksolver # TODO temporary
cm = ConstraintManager(n_wp, [j.name for j in joint_list], fksolver, with_base)
update_fksolver(fksolver, robot_model)

av_start = get_robot_config(robot_model, joint_list, with_base=with_base)
cm.add_eq_configuration(0, av_start)
cm.add_pose_constraint(10, "r_gripper_tool_frame", [1.7, 2.2, 1.0, 0.0, 0.0, 0.0])
cm.add_multi_pose_constraint(11, 
        ["r_gripper_tool_frame", "l_gripper_tool_frame"], 
        [[1.6, 2.1, 1.0, 0.0, 0.0, 0.6], [1.7, 2.2, 1.0]])
cm.add_multi_pose_constraint(12, 
        ["r_gripper_tool_frame", "l_gripper_tool_frame"], 
        [[1.5, 2.0, 1.0, 0.0, 0.0, 0.6], [1.7, 2.2, 1.0]])
cm.add_multi_pose_constraint(13, 
        ["r_gripper_tool_frame", "l_gripper_tool_frame"], 
        [[1.4, 1.9, 1.0, 0.0, 0.0, 0.6], [1.7, 2.2, 1.0]])


av_current = get_robot_config(robot_model, joint_list, with_base=with_base)
av_seq_init = cm.gen_initial_trajectory(av_current)

from pyinstrument import Profiler
profiler = Profiler()
profiler.start()
av_seq = tinyfk_sqp_plan_trajectory(
    sscc, cm, av_seq_init, joint_list, n_wp,
    safety_margin=1e-2, with_base=with_base)
profiler.stop()
print(profiler.output_text(unicode=True, color=True, show_all=True))

viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(641, 480))
viewer.add(robot_model)
viewer.add(fridge)
cv = ConstraintViewer(viewer, cm)
cv.show()
viewer.show()

for av in av_seq:
    set_robot_config(robot_model, joint_list, av, with_base=with_base)
    viewer.redraw()
    time.sleep(1.0)

print('==> Press [q] to close window')
while not viewer.has_exit:
    time.sleep(0.1)
    viewer.redraw()
