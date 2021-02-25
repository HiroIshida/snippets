import copy
import time 
import skrobot
from skrobot.planner.utils import get_robot_config, set_robot_config
from skrobot.model import Box
import numpy as np
import tinyfk
from common import * 

def normalize(vec):
    return vec / np.linalg.norm(vec)

def soft_normalizer(vec):
    return vec / np.linalg.norm(vec)

class RiemannianMotionPolicy(object):
    dim = 3 # for simplicity 
    def __init__(self, f, A):
        # f be function
        self.f = f
        self.A = A

class CollisionRMP(RiemannianMotionPolicy):
    def __init__(self, sdf_): 

        def sdf(x):
            return sdf_(np.array([x])).item()

        def alpha(x):
            dist = sdf(x)
            assert dist > 0.0
            score = np.inf
            a = min(1.0/dist, score)
            return a 

        def weight(x):
            return 1.0/sdf(x)

        def f_acc_inner(x, xd):
            grad = np.zeros(3)
            d0 = sdf(x)
            eps = 1e-6
            for i in range(3):
                x1 = copy.copy(x)
                x1[i] += eps
                grad[i] = (sdf(x1) - d0)/eps
            v_hat = normalize(grad)
            return alpha(x) * v_hat

        def f_acc(x, xd):
            xdd = f_acc_inner(x, xd)
            return xdd

        def A(x, xd):
            xdd_sn = soft_normalizer(f_acc_inner(x, xd))
            return weight(x) * np.outer(xdd_sn, xdd_sn)

        self.f = f_acc
        self.A = A


class TargetRMP(RiemannianMotionPolicy):
    def __init__(self, x_goal):
        def f_acc(x, xd):
            alpha = 5.0
            beta = 40.0
            diff = x_goal - x
            xdd = alpha * soft_normalizer(diff) - beta * xd
            return xdd

        def A(x, xd):
            return np.eye(3)

        super(TargetRMP, self).__init__(f_acc, A)

def pull_back(rmp, jac):

    def A_pullback(x, xd):
        return jac.T.dot(rmp.A(x, xd)).dot(jac)

    def f_acc_pullback(x, xd):
        coef_f_pullback = np.linalg.pinv(A_pullback(x, xd)).dot(jac.T)
        return coef_f_pullback.dot(rmp.f(x, xd)) 
    return RiemannianMotionPolicy(f_acc_pullback, A_pullback)

def sum_rmp(rmp1, rmp2):
    def A_sum(x, xd):
        A1 = rmp1.A(x, xd)
        A2 = rmp2.A(x, xd)
        return A1 + A2

    def f_sum(x, xd):
        A1 = rmp1.A(x, xd)
        A2 = rmp2.A(x, xd)
        tmp =  A1.dot(rmp1.f(x, xd)) + A2.dot(rmp2.f(x, xd))
        return np.linalg.pinv(A1 + A2).dot(tmp)

    return RiemannianMotionPolicy(f_sum, A_sum)

if __name__=='__main__':
    robot_model = skrobot.models.PR2()
    table = Box(extents=[0.6, 0.2, 0.2], with_sdf=True)
    table.translate([0.5, -0.4, 0.8])

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
    viewer.add(table)
    viewer.show()

    x_goal = np.array([0.4, -0.8, 0.7])
    rmp_target = TargetRMP(x_goal)
    rmp_collision = CollisionRMP(table.sdf)

    time.sleep(1.0)


    dt = 5e-2
    for i in range(300):
        P, J = robot_model.fksolver.solve_forward_kinematics(
                [q], [ef_id], joint_ids, 
                with_rot=False, with_base=False, with_jacobian=True, use_cache=False)
        x = P[0]
        x_dot = J.dot(qd)
        rmp_pullback_target = pull_back(rmp_target, J)
        rmp_pullback_collision = pull_back(rmp_collision, J)

        rmp_sum = sum_rmp(rmp_pullback_target, rmp_pullback_collision)
        qdd_sum = rmp_sum.f(x, x_dot)
        qd = qd + qdd_sum * dt
        q = get_robot_config(robot_model, joint_list, with_base=False)
        set_robot_config(robot_model, joint_list, q + qd * dt, with_base=False)

        viewer.redraw()
        time.sleep(0.01)


