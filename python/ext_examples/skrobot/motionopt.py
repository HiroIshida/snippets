import skrobot
import pybullet
import pybullet as pb
import numpy as np
import time 
import copy
from utils import sdf_box
from skrobot.interfaces import PybulletRobotInterface

try:
    robot_model
except:
    robot_model = skrobot.models.urdf.RobotModelFromURDF(urdf_file=skrobot.data.pr2_urdfpath())
    robot_model.init_pose()
    client_id = pybullet.connect(pybullet.GUI)
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI,0)
    viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))
    interface = PybulletRobotInterface(robot_model, connect=client_id)
    interface.angle_vector(robot_model.angle_vector())

    #viewer.add(robot_model)
    #viewer.show()

    link_idx_table = {}
    for link_idx in range(len(robot_model.link_list)):
        name = robot_model.link_list[link_idx].name
        link_idx_table[name] = link_idx

    rarm_end_coords = skrobot.coordinates.CascadedCoords(
            parent=robot_model.r_gripper_tool_frame, 
            name='rarm_end_coords')

    forarm_coords = skrobot.coordinates.CascadedCoords(
            parent=robot_model.r_forearm_link) 

    move_target = rarm_end_coords

    link_names = ["r_shoulder_pan_link", "r_shoulder_lift_link", \
            "r_upper_arm_roll_link", "r_elbow_flex_link", \
            "r_forearm_roll_link", "r_wrist_flex_link", "r_wrist_roll_link"]
    link_list = [robot_model.link_list[link_idx_table[name]] for name in link_names]
    joint_list = [link.joint for link in link_list]
    joint_mimax = [[j.min_angle, j.max_angle] for j in joint_list]

set_joint_angles = lambda av: [j.joint_angle(a) for j, a in zip(joint_list, av)]
get_joint_angles = lambda : np.array([j.joint_angle() for j in joint_list])

def compute_jacobian_skrobot(av0, move_target):
    set_joint_angles(av0)

    base_link = robot_model.link_list[0]
    J = robot_model.calc_jacobian_from_link_list([move_target], link_list, transform_coords=base_link)
    return J, move_target.worldpos()


def create_box(center, b):
    quat = [0, 0, 0, 1]
    vis_id = pb.createVisualShape(pb.GEOM_BOX, halfExtents=b, rgbaColor=[0.0, 1.0, 0, 0.7], physicsClientId=client_id)
    pb.createMultiBody(basePosition=center, baseOrientation=quat, baseVisualShapeIndex=vis_id)
    sdf = sdf_box(b, center) 
    return sdf

def collision_forward_kinematics(av_seq):
    points = []
    jacobs = []
    collision_coords_list = [rarm_end_coords, forarm_coords]
    for av in av_seq:
        for collision_coords in collision_coords_list:
            J, pos = compute_jacobian_skrobot(av, collision_coords)
            points.append(pos)
            jacobs.append(J)
    return np.vstack(points), np.vstack(jacobs)

def endcoord_forward_kinematics(av_seq):
    points = []
    jacobs = []
    for av in av_seq:
        J, pos = compute_jacobian_skrobot(av, rarm_end_coords)
        points.append(pos)
        jacobs.append(J)
    return np.vstack(points), np.vstack(jacobs)

def construct_smoothcost_mat(n_wp):
    acc_block = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]]) 
    vel_block = np.array([[1, -1], [-1, 1]])
    A = np.zeros((n_wp, n_wp))
    for i in [1 + i for i in range(n_wp - 2)]:
        A[i-1:i+2, i-1:i+2] += acc_block # i-1 to i+1 (3x3)
        #A[i-1:i+1, i-1:i+1] += vel_block * 2.0
    return A

def construct_smoothcost_fullmat(n_dof, n_wp, weights = None): # A, b and c terms of chomp B
    w_mat = np.eye(n_dof) if weights is None else np.diag(weights)
    Amat = construct_smoothcost_mat(n_wp)
    Afullmat = np.kron(Amat, w_mat**2)
    return Afullmat

def gen_scipinized(fun):
    closure_member = {'jac_cache': None}
    def fun_scipinized(x):
        f, jac = fun(x)
        closure_member['jac_cache'] = jac
        return f

    fun_scipinized_jac = lambda x: closure_member['jac_cache']
    return fun_scipinized, fun_scipinized_jac

import scipy
class Optimizer:
    def __init__(self, av_seq_init, pos_target, collision_fk, endcoord_fk, joint_limit, sdf, sdf_margin=0.08):
        self.av_seq_init = av_seq_init
        self.n_features = 2
        self.collision_fk = collision_fk
        self.endcoord_fk = endcoord_fk
        self.sdf = lambda X: sdf(X) - sdf_margin
        self.n_wp, self.n_dof = av_seq_init.shape
        self.pos_target = pos_target
        self.joint_limit  = joint_limit
        w = [0.5, 0.5, 0.3, 0.1, 0.1, 0.1, 0.1]
        self.A = construct_smoothcost_fullmat(self.n_dof, self.n_wp, w) 

    def fun_objective(self, x):
        Q = x.reshape(self.n_wp, self.n_dof)
        qe = Q[-1]

        P_ef, J_ef = self.endcoord_fk([qe])
        diff = P_ef[0] - self.pos_target
        cost_terminal = np.linalg.norm(diff) ** 2
        grad_cost_terminal = 2 * diff.dot(J_ef)

        f = (0.5 * self.A.dot(x).dot(x)).item() + cost_terminal
        grad = self.A.dot(x) 
        grad[-self.n_dof:] += grad_cost_terminal # cause only final av affects the terminal cost
        return f, grad

    def fun_ineq(self, xi):
        av_seq = xi.reshape(self.n_wp, self.n_dof)
        P_link, J_link = self.collision_fk(av_seq)

        sdf_grads = np.zeros(P_link.shape)
        F_link_cost0 = self.sdf(np.array(P_link)) 
        eps = 1e-7
        for i in range(3):
            P_link_ = copy.copy(P_link);
            P_link_[:, i] += eps;
            F_link_cost1 = self.sdf(np.array(P_link_))  
            sdf_grads[:, i] = (F_link_cost1 - F_link_cost0)/eps;

        sdf_grads = sdf_grads.reshape(self.n_wp * self.n_features, 1, 3)
        J_link = J_link.reshape(self.n_wp * self.n_features, 3, self.n_dof)
        J_link_list = np.matmul(sdf_grads, J_link)
        J_link_block = J_link_list.reshape(self.n_wp, self.n_features, self.n_dof) #(n_wp, n_features, n_dof)
        J_link_full = scipy.linalg.block_diag(*list(J_link_block)) #(n_wp * n_features, n_wp * n_dof)
        F_cost_full, J_cost_full = F_link_cost0, J_link_full
        return F_cost_full, J_cost_full

    def fun_eq(self, xi):
        # terminal constraint
        Q = xi.reshape(self.n_wp, self.n_dof)
        qs = Q[0]
        q_start = self.av_seq_init[0]
        f = np.hstack((q_start - qs))
        grad_ = np.zeros((self.n_dof * 1, self.n_dof * self.n_wp))
        grad_[0:self.n_dof, 0:self.n_dof] = - np.eye(self.n_dof) 
        return f, grad_

    def solve(self):
        eq_const_scipy, eq_const_jac_scipy = gen_scipinized(self.fun_eq)
        eq_dict = {'type': 'eq', 'fun': eq_const_scipy, 'jac': eq_const_jac_scipy}
        ineq_const_scipy, ineq_const_jac_scipy = gen_scipinized(self.fun_ineq)
        ineq_dict = {'type': 'ineq', 'fun': ineq_const_scipy, 'jac': ineq_const_jac_scipy}
        f, jac = gen_scipinized(self.fun_objective)

        tmp = np.array(self.joint_limit)
        lower_limit = tmp[:, 0]
        uppre_limit = tmp[:, 1]

        bounds = list(zip(lower_limit, uppre_limit)) * self.n_wp

        xi_init = self.av_seq_init.reshape((self.n_dof * self.n_wp, ))
        res = scipy.optimize.minimize(f, xi_init, method='SLSQP', jac=jac,
                bounds = bounds,
                constraints=[eq_dict, ineq_dict], options={'ftol': 1e-4, 'disp': False})
        print(res)
        traj_opt = res.x.reshape(self.n_wp, self.n_dof)
        return traj_opt

set_joint_angles([0]*7)
interface.angle_vector(robot_model.angle_vector())

target_coords = skrobot.coordinates.Coordinates([0.5, 0.5, 0.8], [0, 0, 0])
res = robot_model.inverse_kinematics(
        target_coords, link_list=link_list, move_target=move_target, rotation_axis=False)
av_init = get_joint_angles()
set_joint_angles(av_init)
print("setting")
time.sleep(1.0)
interface.angle_vector(robot_model.angle_vector())

target_coords = skrobot.coordinates.Coordinates([0.8, -0.5, 0.8], [0, 0, 0])
res = robot_model.inverse_kinematics(
        target_coords, link_list=link_list, move_target=move_target, rotation_axis=False)
av_target = get_joint_angles()
n_wp = 8
step = (av_target - av_init)/(n_wp - 1)
init_trajectory = np.array([av_init + i * step for i in range(n_wp)])
sdf = create_box([0.9, -0.2, 0.9], [0.4, 0.25, 0.3])
opt = Optimizer(init_trajectory, [0.7, -0.7, 1.0], collision_forward_kinematics, endcoord_forward_kinematics, joint_mimax, sdf)
sol_trajectory = opt.solve()


time.sleep(1.0)
print("show trajectory")
for av in sol_trajectory:
    set_joint_angles(av)
    interface.angle_vector(robot_model.angle_vector())

    P_link, J_link = collision_forward_kinematics([av])
    print("debug")
    print(P_link[0].reshape((-1, 3)))
    sd1 = sdf(P_link[0].reshape((-1, 3)))
    print(sd1)
    time.sleep(0.5)
