import time 
import skrobot
from skrobot.planner.utils import get_robot_config, set_robot_config
import numpy as np
import tinyfk


from common import * 

def rmp_target(x_goal, x, xd):
    alpha = 5.0
    beta = 40.0
    diff = x_goal - x
    xdd = alpha * diff/np.linalg.norm(diff) - beta * xd
    return xdd

if __name__=='__main__':
    robot_model = skrobot.models.PR2()
    #fksolver = tinyfk.RobotModel(robot_model.urdf_path)
    coll_link_list = rarm_coll_link_list(robot_model)
    joint_list = rarm_joint_list(robot_model)
    joint_names = [j.name for j in joint_list]

    joint_ids = robot_model.fksolver.get_joint_ids(joint_names)
    collink_ids = robot_model.fksolver.get_link_ids([l.name for l in coll_link_list])
    ef_id = robot_model.fksolver.get_link_ids(["r_gripper_tool_frame"])[0]
    q = get_robot_config(robot_model, joint_list, with_base=False)
    qd = q * 0.0

    viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(641, 480))
    viewer.add(robot_model)
    viewer.show()

    x_goal = np.array([0.7, -0.5, 0.7])
    dt = 1e-1
    for i in range(100):
        P, J = robot_model.fksolver.solve_forward_kinematics(
                [q], [ef_id], joint_ids, 
                with_rot=False, with_base=False, with_jacobian=True, use_cache=False)
        print(P)
        x_dot = J.dot(qd)
        xdd = rmp_target(x_goal, P[0], x_dot)
        n_dof = J.shape[1]
        qdd = np.linalg.inv(J.T.dot(J) + np.eye(n_dof)).dot(J.T).dot(xdd)

        qd = qd + qdd * dt
        q = get_robot_config(robot_model, joint_list, with_base=False)
        set_robot_config(robot_model, joint_list, q + qd * dt, with_base=False)

        viewer.redraw()
        time.sleep(0.1)
