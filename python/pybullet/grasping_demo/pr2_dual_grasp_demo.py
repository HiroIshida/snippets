from math import *
import pybullet 
import pybullet as pb
import pybullet_data
import numpy as np
import time
import utils

class RGripper:
    def __init__(self):

        def construct_tables(robot):
            # construct tables
            # table (joint_name -> id) (link_name -> id)
            link_table = {pb.getBodyInfo(robot, physicsClientId=CLIENT)[0].decode('UTF-8'):-1}
            joint_table = {}
            heck = lambda path: "_".join(path.split("/"))
            for _id in range(pb.getNumJoints(robot, physicsClientId=CLIENT)):
                joint_info = pb.getJointInfo(robot, _id, physicsClientId=CLIENT)
                joint_id = joint_info[0]

                joint_name = joint_info[1].decode('UTF-8')
                joint_table[joint_name] = joint_id
                name_ = joint_info[12].decode('UTF-8')
                name = heck(name_)
                link_table[name] = _id
            return link_table, joint_table

        self.gripper = pb.loadURDF("pr2_description/urdf/pr2_rgripper.urdf", useFixedBase=True)
        link_table, joint_table = construct_tables(self.gripper)
        self.link_table = link_table
        self.joint_table = joint_table

    def get_state(self):
        angle, velocity, _, _ = pb.getJointState(self.gripper, 0)
        return angle, velocity

    def set_gripper_width(self, angle, force=False):
        """
        if force, angle is set regardless of the physics
        """
        if force:
            for i in [8,9,10,11]: 
                pb.resetJointState(self.gripper, i, angle,
                        targetVelocity = 0.0)
        else:
            for i in [8,9,10,11]: 
                pb.setJointMotorControl2(self.gripper, i, 
                        pb.POSITION_CONTROL, targetPosition=angle, force=300)

    def set_state(self, state):
        for i in range(3):
            pb.setJointMotorControl2(self.gripper, i, 
                    pb.POSITION_CONTROL, targetPosition=state[i], force=300)

    def set_basepose(self, pos, rpy):
        utils.set_6dpose(self.gripper, pos, rpy)

    def set_pose(self, rpy):
        quat = utils.quat_from_euler(rpy)
        utils.set_quat(self.gripper, quat)

class LGripper:
    def __init__(self):

        def construct_tables(robot):
            # construct tables
            # table (joint_name -> id) (link_name -> id)
            link_table = {pb.getBodyInfo(robot, physicsClientId=CLIENT)[0].decode('UTF-8'):-1}
            joint_table = {}
            heck = lambda path: "_".join(path.split("/"))
            for _id in range(pb.getNumJoints(robot, physicsClientId=CLIENT)):
                joint_info = pb.getJointInfo(robot, _id, physicsClientId=CLIENT)
                joint_id = joint_info[0]

                joint_name = joint_info[1].decode('UTF-8')
                joint_table[joint_name] = joint_id
                name_ = joint_info[12].decode('UTF-8')
                name = heck(name_)
                link_table[name] = _id
            return link_table, joint_table

        self.gripper = pb.loadURDF("pr2_description/urdf/pr2_lgripper.urdf", useFixedBase=True)
        link_table, joint_table = construct_tables(self.gripper)
        self.link_table = link_table
        self.joint_table = joint_table

    def get_state(self):
        angle, velocity, _, _ = pb.getJointState(self.gripper, 0)
        return angle, velocity

    def set_gripper_width(self, angle, force=False):
        """
        if force, angle is set regardless of the physics
        """
        if force:
            for i in [8,9,10,11]:
                pb.resetJointState(self.gripper, i, angle,
                        targetVelocity = 0.0)
        else:
            for i in [8,9,10,11]:     
                pb.setJointMotorControl2(self.gripper, i, 
                        pb.POSITION_CONTROL, targetPosition=angle, force=300)

    def set_state(self, state):
        for i in range(3):
            pb.setJointMotorControl2(self.gripper, i, 
                    pb.POSITION_CONTROL, targetPosition=state[i], force=300)

    def set_basepose(self, pos, rpy):
        utils.set_6dpose(self.gripper, pos, rpy)

    def set_pose(self, rpy):
        quat = utils.quat_from_euler(rpy)
        utils.set_quat(self.gripper, quat)
try:
    isInit
except:
    isInit = True
    CLIENT = pybullet.connect(pybullet.GUI)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF
    plane = pybullet.loadURDF("plane.urdf")
    pb.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0, physicsClientId=CLIENT)
    pb.configureDebugVisualizer(pybullet.COV_ENABLE_TINY_RENDERER, 0, physicsClientId=CLIENT)
    pb.setGravity(0,0,-10.0)
    table = pb.loadURDF("table/table.urdf", physicsClientId=CLIENT)
    plate = pb.loadURDF("dish/plate.urdf", physicsClientId=CLIENT)
    rgripper = RGripper()
    lgripper = LGripper()

table_pos = np.array([0.0, 0.0, 0.0])
utils.set_point(table, table_pos)
utils.set_zrot(table, pi*0.5)

utils.set_point(plate, [0.0, 0.0, 0.63])
rgripper.set_basepose([0, 0.25, 0.78], [-1.54, 0.6, -1.57])
lgripper.set_basepose([0, -0.23, 0.77], [1.54, 0.65, 1.57])
rgripper.set_gripper_width(0.5, force=True)
lgripper.set_gripper_width(0.5, force=True)

time.sleep(7)
rgripper.set_gripper_width(0.0)
rgripper.set_state([-0.2, 0.5, -0.01])
lgripper.set_state([0, 0, 0.5])
rgripper.set_pose([-1.54, 0.8, -1.57])

for i in range(100):
    pb.stepSimulation(physicsClientId=CLIENT)
    time.sleep(0.005)

rgripper.set_state([0.0, -0.5, 0.0])
lgripper.set_state([0.0, -0.5, 0.1])
for i in range(1000):
    pb.stepSimulation(physicsClientId=CLIENT)
    time.sleep(0.005)
