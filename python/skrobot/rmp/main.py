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

class RiemannianMotionPolicy(object):
    dim = 3 # for simplicity 
    def __init__(self, f, A=None):
        # f be function
        if A is None:
            A = np.eye(self.dim)
        self.f = f
        self.A = A

class TargetRMP(RiemannianMotionPolicy):
    def __init__(self, x_goal):
        def f_acc(x, xd):
            alpha = 5.0
            beta = 40.0
            diff = x_goal - x
            xdd = alpha * diff/np.linalg.norm(diff) - beta * xd
            return xdd
        super(TargetRMP, self).__init__(f_acc)

def pull_back(rmp, jac):
    A_pullback = jac.T.dot(rmp.A).dot(jac)
    def f_acc_pullback(x, xd):
        coef_f_pullback = np.linalg.pinv(A_pullback).dot(jac.T)
        return coef_f_pullback.dot(rmp.f(x, xd))
    return RiemannianMotionPolicy(f_acc_pullback, A_pullback)

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

    x_goal = np.array([0.7, -0.7, 0.7])
    rmp = TargetRMP(x_goal)

    dt = 5e-2
    for i in range(100):
        P, J = robot_model.fksolver.solve_forward_kinematics(
                [q], [ef_id], joint_ids, 
                with_rot=False, with_base=False, with_jacobian=True, use_cache=False)
        x = P[0]
        x_dot = J.dot(qd)
        rmp_pullback = pull_back(rmp, J)
        qdd = rmp_pullback.f(x, x_dot)

        qd = qd + qdd * dt
        q = get_robot_config(robot_model, joint_list, with_base=False)
        set_robot_config(robot_model, joint_list, q + qd * dt, with_base=False)

        viewer.redraw()
        time.sleep(0.03)
