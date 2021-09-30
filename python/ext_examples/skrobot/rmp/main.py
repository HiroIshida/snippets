import copy
import time 
import skrobot
from skrobot.planner.utils import get_robot_config, set_robot_config
from skrobot.model import Box
import numpy as np
import tinyfk
import common
from common import * 

def normalize(vec):
    return vec / np.linalg.norm(vec)

def soft_normalizer(vec):
    return vec / np.linalg.norm(vec)

def pull_back(rmp, jac):

    def A_pullback(x, xd):
        return jac.T.dot(rmp.A(x, xd)).dot(jac)

    def f_acc_pullback(x, xd):
        coef_f_pullback = np.linalg.pinv(A_pullback(x, xd)).dot(jac.T)
        return coef_f_pullback.dot(rmp.f(x, xd)) 
    return RiemannianMotionPolicy(f_acc_pullback, A_pullback)


class RiemannianMotionPolicy(object):
    dim = 3 # for simplicity 
    def __init__(self, f, A):
        # f be function
        self.f = f
        self.A = A

class RobotCollisionRMP(RiemannianMotionPolicy):
    """
    with respect to the configuration space
    """

    def __init__(self, sdf, with_base=False):
        # delete dependency on skrobot PR2()
        robot = skrobot.models.PR2()

        #urdf_path = skrobot.data.pr2_urdfpath()
        self.joint_list = common.rarm_joint_list(robot)
        self.coll_link_list = common.rarm_coll_link_list(robot)

        # start initializing 
        self.with_base = False
        self.fksolver = robot.fksolver
        self.joint_ids = robot.fksolver.get_joint_ids([j.name for j in self.joint_list])
        self.coll_link_ids = robot.fksolver.get_link_ids([l.name for l in self.coll_link_list])

        self.rmp_collision = CollisionRMP(sdf)

    def resolve(self, q, qd):
        points, jacs_tmp = robot_model.fksolver.solve_forward_kinematics(
                [q], self.coll_link_ids, self.joint_ids, 
                with_rot=False, with_base=self.with_base, with_jacobian=True, use_cache=False)

        n_coll_points = len(points) 
        n_joint = len(self.joint_ids)

        jacs = jacs_tmp.reshape(n_coll_points, 3, n_joint)

        A_list = []
        f_list = []
        for x, jac in zip(points, jacs):
            rmp_pb = pull_back(self.rmp_collision, jac)
            xd = jac.dot(qd)
            f = rmp_pb.f(x, xd)
            A = rmp_pb.A(x, xd)

            A_list.append(A)
            f_list.append(f)

        A_sum = sum(A_list)
        Af_sum = sum([A.dot(f) for A, f in zip(A_list, f_list)])

        A_merged = A_sum
        f_merged = np.linalg.pinv(A_sum).dot(Af_sum)
        return f_merged, A_merged


class CollisionRMP(RiemannianMotionPolicy):
    def __init__(self, sdf_): 

        def sdf(x):
            return sdf_(np.array([x])).item()

        def alpha(x):
            dist = sdf(x)

            cutoff = 0.01
            minimum = 1e-8
            a = min(max(cutoff - (cutoff * 10.0) * dist, minimum), cutoff)
            a = 1e-8
            return a 

        def weight(x):
            #return alpha(x)
            #return 1.0/sdf(x)
            return 1e-8

        def f_acc_inner(x, xd):
            grad = np.zeros(3)
            d0 = sdf(x)
            eps = 1e-6
            for i in range(3):
                x1 = copy.copy(x)
                x1[i] += eps
                grad[i] = (sdf(x1) - d0)/eps
            v_hat = normalize(grad)
            return alpha(x) * v_hat# - beta(x) * np.outer(v_hat, v_hat).dot(xd)

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
            alpha = 10.0
            beta = 20.0
            diff = x_goal - x
            xdd = alpha * soft_normalizer(diff) - beta * xd
            return xdd

        def A(x, xd):
            return np.eye(3)

        super(TargetRMP, self).__init__(f_acc, A)

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
    table = Box(extents=[0.6, 0.1, 0.1], with_sdf=True)
    table.translate([0.5, -0.4, 1.4])

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
    rmp_rc = RobotCollisionRMP(table.sdf)

    time.sleep(1.0)


    dt = 1e-2
    for i in range(10000):
        print(i)
        P, J = robot_model.fksolver.solve_forward_kinematics(
                [q], [ef_id], joint_ids, 
                with_rot=False, with_base=False, with_jacobian=True, use_cache=False)
        x = P[0]
        x_dot = J.dot(qd)
        rmp_pullback_target = pull_back(rmp_target, J)
        rmp_pullback_collision = pull_back(rmp_collision, J)

        ts = time.time()
        f1, A1 = rmp_rc.resolve(q, qd)
        f2 = rmp_pullback_target.f(x, x_dot)
        A2 = rmp_pullback_target.A(x, x_dot)
        f_whole = np.linalg.pinv(A1 + A2).dot(A1.dot(f1) + A2.dot(f2))
        #print(time.time() - ts)

        rmp_sum = sum_rmp(rmp_pullback_target, rmp_pullback_collision)
        qdd_sum = f_whole
        qd = qd + qdd_sum * dt
        q = get_robot_config(robot_model, joint_list, with_base=False)
        set_robot_config(robot_model, joint_list, q + qd * dt, with_base=False)

        viewer.redraw()
        time.sleep(0.01)


