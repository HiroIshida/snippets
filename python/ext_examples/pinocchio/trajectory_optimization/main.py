import time
import numpy as np
from robot_descriptions.loaders.pinocchio import load_robot_description
import pinocchio as pin

robot: pin.RobotWrapper = load_robot_description("jaxon_description")
q = np.zeros(robot.nq)
v = np.zeros(robot.nv)  # -1 dimension as q because quaternion parameterization stuff

for joint_id in range(1, robot.model.njoints):  # Joint 0 is the universe joint
    joint = robot.model.joints[joint_id]
    joint_name = robot.model.names[joint_id]
    print(joint_name)
    print(joint)
    # you must remove continuous joints!!!! probably because it use cos and sin to parameterize the joint. so 2 > 1
    assert joint.nq == joint.nv

T = 20
varname_range_table = {}
head = 0
ts = time.time()
for i in range(T):
    varname_range_table[f"q{i}"] = (head, head + robot.nq)
    head += robot.nq
    varname_range_table[f"v{i}"] = (head, head + robot.nv)
    head += robot.nv
    varname_range_table[f"r{i}"] = (head, head + 3)
    head += 3
    varname_range_table[f"rd{i}"] = (head, head + 3)
    head += 3
    varname_range_table[f"rdd{i}"] = (head, head + 3)
    head += 3

    # left foot contact force
    for j in range(4):
        varname_range_table[f"Fl{j}_{i}"] = (head, head + 3)
        head += 3
    # right foot contact force
    for j in range(4):
        varname_range_table[f"Fr{j}_{i}"] = (head, head + 3)
        head += 3

    varname_range_table[f"h{i}"] = (head, head + 6)
    head += 6
    varname_range_table[f"hd{i}"] = (head, head + 6)
    head += 6
print("Time elapsed:", time.time() - ts)

constname_range_table = {}
head = 0
ts = time.time()
for i in range(T):
    constname_range_table["momentum_eq_const" + str(i)] = (head, head + 3)
    head += 3
    constname_range_table["angular_momentum_eq_const" + str(i)] = (head, head + 3)
    head += 3
    constname_range_table["cmm_eq_const" + str(i)] = (head, head + 6)
    head += 6
    constname_range_table["q_euler_eq_const" + str(i)] = (head, head + robot.nq)


mass = 100


