from tempfile import TemporaryDirectory
import trimesh
import time
import pybullet
from skrobot.models import PR2
from skrobot.interfaces import PybulletRobotInterface
from skrobot.models import PR2
from skrobot.coordinates import Coordinates
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union, List
from xml.etree.ElementTree import Element, ElementTree, SubElement

import numpy as np
from skrobot.model.primitives import Box
from skrobot.coordinates import Coordinates
from skrobot.coordinates.math import rpy2quaternion, wxyz2xyzw
from skmp.robot.pr2 import PR2Config
from skmp.constraint import PoseConstraint, CollFreeConst
from skmp.solver.interface import Problem
from skmp.robot.utils import get_robot_state, set_robot_state
from skmp.satisfy import SatisfactionResult, satisfy_by_optimization, satisfy_by_optimization_with_budget




def load_mesh(mesh_path: Path, scale: float = 1.0) -> int:
    tmp_urdf_file = """
        <?xml version="1.0" ?>
        <robot name="tmp">
        <link name="base_link" concave="yes">
        <visual>
        <geometry>
            <mesh filename="{mesh_path}" scale="{scale} {scale} {scale}"/>
        </geometry>
        <material name="">
          <color rgba="0.6 0.6 0.6 1.0" />
        </material>
        </visual>
        <collision concave="yes">
        <geometry>
        <mesh filename="{mesh_path}" scale="{scale} {scale} {scale}"/>
        </geometry>
        </collision>
        </link>
        </robot>
    """.format(mesh_path=str(mesh_path), scale=scale)
    # save urdf file to temporary file
    with TemporaryDirectory() as td:
        urdf_file_path = Path(td) / "tmp.urdf"
        with open(urdf_file_path, "w") as f:
            f.write(tmp_urdf_file)
        obj_id = pybullet.loadURDF(str(urdf_file_path))
    return obj_id


class PR2PybulletRobotInterface(PybulletRobotInterface):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.force = 500.0

    def move_gripper(self, angle, simulate=False, arm="rarm"):
        assert self.robot_id is not None
        joint_names: List[str]
        if arm == "rarm":
            joint_names = ['r_gripper_l_finger_joint', 'r_gripper_l_finger_tip_joint', 'r_gripper_r_finger_joint', 'r_gripper_r_finger_tip_joint']
        elif arm == "larm":
            joint_names = ['l_gripper_l_finger_joint', 'l_gripper_l_finger_tip_joint', 'l_gripper_r_finger_joint', 'l_gripper_r_finger_tip_joint']

        for joint_name in joint_names:
            idx = self.joint_name_to_joint_id[joint_name]

            if simulate:
                pybullet.setJointMotorControl2(bodyIndex=self.robot_id,
                                        jointIndex=idx,
                                        controlMode=pybullet.POSITION_CONTROL,
                                        targetPosition=angle,
                                        targetVelocity=self.target_velocity,
                                        force=self.force,
                                        positionGain=self.position_gain,
                                        velocityGain=self.velocity_gain,
                                        maxVelocity=self.max_velocity)
            else:
                pybullet.resetJointState(self.robot_id, idx, angle)

    def wait_interpolation(self, thresh=0.05, timeout=60.0, callback=None):
        ignore_joint_names = ["l_gripper_motor_screw_joint", "r_gripper_motor_screw_joint"]
        ignore_joint_ids = [self.joint_name_to_joint_id[name] for name in ignore_joint_names]
        joint_id_to_name = {v: k for k, v in self.joint_name_to_joint_id.items()}
        torso_idx = self.joint_name_to_joint_id["torso_lift_joint"]

        start = time.time()
        while True:
            pybullet.stepSimulation()
            wait = False
            for idx in self.joint_ids:
                if idx in ignore_joint_ids:
                    continue
                if idx is None:
                    continue
                _, velocity, _, _ = pybullet.getJointState(self.robot_id,
                                                    idx)
                if abs(velocity) > thresh:
                    name = joint_id_to_name[idx]
                    # print(f"{name} is moving with velocity: {velocity}")
                    wait = True
            if wait is False:
                break
            if time.time() - start > timeout:
                return False

            if callback is not None:
                callback(self)

        return True

    def angle_vector(self, angle_vector=None, realtime_simulation=None):
        torso_idx = self.joint_name_to_joint_id["torso_lift_joint"]

        if realtime_simulation is not None and isinstance(
                realtime_simulation, bool):
            self.realtime_simulation = realtime_simulation

        if self.robot_id is None:
            return self.robot.angle_vector()
        if angle_vector is None:
            angle_vector = self.robot.angle_vector()

        for i, (joint, angle) in enumerate(
                zip(self.robot.joint_list, angle_vector)):
            idx = self.joint_name_to_joint_id[joint.name]
            if idx == torso_idx:
                force = 20000
            else:
                force = self.force

            joint = self.robot.joint_list[i]

            if self.realtime_simulation is False:
                pybullet.resetJointState(self.robot_id, idx, angle)

            pybullet.setJointMotorControl2(bodyIndex=self.robot_id,
                                    jointIndex=idx,
                                    controlMode=pybullet.POSITION_CONTROL,
                                    targetPosition=angle,
                                    targetVelocity=self.target_velocity,
                                    force=force,
                                    positionGain=self.position_gain,
                                    velocityGain=self.velocity_gain,
                                    maxVelocity=self.max_velocity)

        return angle_vector


def create_debug_axis(coords: Coordinates, length: float = 0.1):
    start = coords.worldpos()
    end_x = start + coords.rotate_vector([length, 0, 0])
    pybullet.addUserDebugLine(start, end_x, [1, 0, 0], 3)
    end_y = start + coords.rotate_vector([0, length, 0])
    pybullet.addUserDebugLine(start, end_y, [0, 1, 0], 3)
    end_z = start + coords.rotate_vector([0, 0, length])
    pybullet.addUserDebugLine(start, end_z, [0, 0, 1], 3)


class Environment:
    pr2: PR2
    ri: PR2PybulletRobotInterface
    client_id: int
    table_id: int
    cup_id: int
    co_handle: Coordinates
    co_grasp_pre: Coordinates
    co_grasp: Coordinates

    def __init__(self, gui: bool = False):
        pr2 = PR2()
        pr2.reset_manip_pose()
        pr2.torso_lift_joint.joint_angle(0.3)
        pr2.r_shoulder_lift_joint.joint_angle(-0.3)
        pr2.l_shoulder_lift_joint.joint_angle(-0.3)
        if gui:
            client_id = pybullet.connect(pybullet.GUI)
        else:
            client_id = pybullet.connect(pybullet.DIRECT)
        pybullet.setGravity(0, 0, -9.8)
        ri = PR2PybulletRobotInterface(pr2, use_fixed_base=True, connect=client_id)
        ri.angle_vector(pr2.angle_vector(), realtime_simulation=False)
        ri.move_gripper(0.3, simulate=False, arm="rarm")
        ri.sync()
        ri.wait_interpolation()
        self.pr2 = pr2
        self.ri = ri
        self.client_id = client_id

        # create box
        box_size = np.array([0.6, 0.8, 0.8])
        box_pos = [0.8, 0.0, 0.4]
        vis_id = pybullet.createVisualShape(pybullet.GEOM_BOX, halfExtents=0.5 * box_size)
        col_id = pybullet.createCollisionShape(pybullet.GEOM_BOX, halfExtents=0.5 * box_size)
        self.table_id = pybullet.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vis_id, basePosition=box_pos)

        # create cup
        self.cup_id = load_mesh("./cup_reduce.obj", scale=0.03)
        pybullet.changeDynamics(self.cup_id, -1, lateralFriction=1.5)
        q = rpy2quaternion([np.pi / 2, 0, 0])
        pybullet.resetBasePositionAndOrientation(self.cup_id, [0.6, 0.0, 0.8], q)

        pybullet.setTimeStep(1.0 / 1000.)

        # define handle pose
        pos, quat = pybullet.getBasePositionAndOrientation(self.cup_id)
        co_handle = Coordinates(pos, quat)
        co_handle.translate([0, 0, 0.07])
        co_handle.rotate(np.pi * 0.2, "z")
        co_handle.translate([-0.12, -0.0, 0.0])
        co_handle.rotate(-np.pi * 0.5, "z")
        create_debug_axis(co_handle)

        # define aux poses
        co_grasp = co_handle.copy_worldcoords()
        co_grasp.rotate(0.3, "z")
        co_grasp.translate([0.02, -0.015, 0.0])

        co_grasp_pre = co_grasp.copy_worldcoords()
        co_grasp_pre.translate([-0.06, -0.0, 0.0])

        create_debug_axis(co_grasp_pre)
        create_debug_axis(co_grasp)
        self.co_handle = co_handle
        self.co_grasp = co_grasp
        self.co_grasp_pre = co_grasp_pre

    def solve_ik(self, co: Coordinates, simulate: bool = False, random_sampling: bool = False):
        pr2_conf = PR2Config()
        colkin = pr2_conf.get_collision_kin()
        efkin = pr2_conf.get_endeffector_kin()
        efkin.reflect_skrobot_model(self.pr2)
        colkin.reflect_skrobot_model(self.pr2)

        obstacle = Box([0.6, 0.8, 0.8], pos=[0.8, 0.0, 0.4], with_sdf=True)

        box_const = pr2_conf.get_box_const()
        collfree_const = CollFreeConst(colkin, obstacle.sdf, self.pr2)
        goal_eq_const = PoseConstraint.from_skrobot_coords([co], efkin, self.pr2)
        joint_list = pr2_conf._get_control_joint_names()
        q_start = get_robot_state(self.pr2, joint_list)
        if random_sampling:
            res = satisfy_by_optimization_with_budget(goal_eq_const, box_const, collfree_const, q_start)
        else:
            res = satisfy_by_optimization(goal_eq_const, box_const, collfree_const, q_start)

        if not res.success:
            return False

        set_robot_state(self.pr2, joint_list, res.q)
        self.ri.angle_vector(self.pr2.angle_vector(), realtime_simulation=simulate)
        self.ri.wait_interpolation()
        return True

    def translate(self, trans: np.ndarray, simulate: bool = False, random_sampling: bool = False) -> bool:
        co = self.pr2.rarm_end_coords.copy_worldcoords()
        co.translate(trans) 
        return self.solve_ik(co, simulate=simulate, random_sampling=random_sampling)

    def grasp(self, simulate: bool = False):
        angles = np.linspace(0.3, 0.05, 20)
        for angle in angles:
            print(angle)
            self.ri.move_gripper(angle, simulate=simulate)
            self.ri.sync()
            self.ri.wait_interpolation()

