import time
import numpy as np
import vamp
import pandas as pd

(vamp_module, planner_func, plan_settings,
 simp_settings) = vamp.configure_robot_and_planner_with_kwargs("fetch", "rrtc")
e = vamp.Environment()
table = vamp.Cuboid([0.9, 0.0, 0.8], [0, 0, 0], np.array([1.0, 2.0, 0.05]) * 0.5)
ground = vamp.Cuboid([0.0, 0.0, -0.1], [0, 0, 0], np.array([2.0, 2.0, 0.05]) * 0.5)
e.add_cuboid(table)
e.add_cuboid(ground)

start = [ 0.,          1.31999949,  1.40000015, -0.20000077,  1.71999929,  0., 1.6600001,   0.        ]
goal = [ 0.38625,     0.20565826,  1.41370123,  0.30791941, -1.82230466,  0.24521043, 0.41718824,  6.01064401]
assert vamp.fetch.validate(start, e)
assert vamp.fetch.validate(goal, e)
ts = time.time()
n_repeat = 1000
for _ in range(n_repeat):
    result = planner_func(start, goal, e, plan_settings)
    simple = vamp_module.simplify(result.path, e, simp_settings)
print(f"average planning time: {(time.time() - ts) / n_repeat} sec")

# visualize using skrobot
from skrobot.models.fetch import Fetch
from skrobot.viewers import PyrenderViewer
from skrobot.planner.utils import set_robot_config
from skrobot.model.primitives import Axis, Box

joint_names = [
    "torso_lift_joint",
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "upperarm_roll_joint",
    "elbow_flex_joint",
    "forearm_roll_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
]
fetch = Fetch()
joint_list = [fetch.__dict__[name] for name in joint_names]

table = Box([1.0, 2.0, 0.05], with_sdf=True)
table.translate([1.0, 0.0, 0.8])

v = PyrenderViewer()
v.add(fetch)
v.add(table)
v.show()
time.sleep(2)

for q in simple.path:
  set_robot_config(fetch, joint_list, np.array(q.to_list()))
  v.redraw()
  time.sleep(0.2)
