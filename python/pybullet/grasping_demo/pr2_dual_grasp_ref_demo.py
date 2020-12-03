#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from math import *
import pybullet 
import pybullet as pb
import pybullet_data
import numpy as np
import time
import six
import utils
from grasp_utils import *

class RGripper():
    def __init__(self):

        def construct_tables(robot):
            # construct tables
            # table (joint_name -> id) (link_name -> id)
            link_table = {pb.getBodyInfo(robot)[0].decode('UTF-8'):-1}
            joint_table = {}
            heck = lambda path: "_".join(path.split("/"))
            for _id in range(pb.getNumJoints(robot)):
                joint_info = pb.getJointInfo(robot, _id)
                joint_id = joint_info[0]

                joint_name = joint_info[1].decode('UTF-8')
                joint_table[joint_name] = joint_id
                name_ = joint_info[12].decode('UTF-8')
                name = heck(name_)
                link_table[name] = _id
            return link_table, joint_table

        self.gripper = pb.loadURDF("pr2_description/urdf/pr2_rgripper.urdf", useFixedBase=True)
        print("Rgripper ID", self.gripper)
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

    def set_angle(self, angle, force=False):
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
                        pb.POSITION_CONTROL, targetPosition=angle, force=200)
class LGripper():
    def __init__(self):

        def construct_tables(robot):
            # construct tables
            # table (joint_name -> id) (link_name -> id)
            link_table = {pb.getBodyInfo(robot)[0].decode('UTF-8'):-1}
            joint_table = {}
            heck = lambda path: "_".join(path.split("/"))
            for _id in range(pb.getNumJoints(robot)):
                joint_info = pb.getJointInfo(robot, _id)
                joint_id = joint_info[0]

                joint_name = joint_info[1].decode('UTF-8')
                joint_table[joint_name] = joint_id
                name_ = joint_info[12].decode('UTF-8')
                name = heck(name_)
                link_table[name] = _id
            return link_table, joint_table

        self.gripper = pb.loadURDF("pr2_description/urdf/pr2_lgripper.urdf", useFixedBase=True)
        print("Lgripper ID", self.gripper)
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

    def set_angle(self, angle, force=False):
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
                        pb.POSITION_CONTROL, targetPosition=angle, force=200)
class Dish:
    def __init__(self):
        self.dish = pb.loadURDF("dish/plate.urdf")

class Simulator(object):
    def __init__(self):
        try:
            isInit
        except:
            isInit = True
            CLIENT = pybullet.connect(pybullet.GUI)
            print("client",CLIENT)
            pb.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF
            plane = pybullet.loadURDF("plane.urdf")
            print("plane ID", plane)
            #pb.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0, physicsClientId=CLIENT)
            #pb.configureDebugVisualizer(pybullet.COV_ENABLE_TINY_RENDERER, 0, physicsClientId=CLIENT)
            pb.setGravity(0,0,-10.0)
            self.table = pb.loadURDF("table/table.urdf")
            print("table ID", self.table)
            self.plate = pb.loadURDF("dish/plate.urdf")
            print("plate ID", self.plate)
            self.rgripper = RGripper()
            self.lgripper = LGripper()
        self.try_num = 5
        self.frames = []
        pb.resetDebugVisualizerCamera(2.0, 90, -0.0, (0.0, 0.0, 1.0))
        
        self.viewMatrix = pb.computeViewMatrix(
            cameraEyePosition=[5, 5, 30],
            cameraTargetPosition=[3, 3, 3],
            cameraUpVector=[3, 1, 3])
        self.projectionMatrix = pb.computeProjectionMatrixFOV(
            fov=45.0,
            aspect=1.0,
            nearVal=0.1,
            farVal=3.1) 
    
    def reset(self):
        table_pos = np.array([0.0, 0.0, 0.0])
        utils.set_point(self.table, table_pos)
        utils.set_zrot(self.table, pi*0.5)
        utils.set_point(self.plate, [0.0, 0.0, 0.63])
        self.rgripper.set_basepose([0, 0.25, 0.78], [-1.54, 0.6, -1.57])
        self.rgripper.set_state([0, 0, 0])
        self.rgripper.set_angle(self.rgripper.gripper, 0)
        self.lgripper.set_basepose([0, -0.23, 0.77], [1.54, 0.65, 1.57])
        self.lgripper.set_state([0, 0, 0])
        self.lgripper.set_angle(self.lgripper.gripper, 0)
        self.rgripper.set_gripper_width(0.5, force=True)
        self.lgripper.set_gripper_width(0.5, force=True)
        for i in range(100):
            pb.stepSimulation()

    def rollout(self):
        try:
            for try_count in six.moves.range(self.try_num):
                self.reset()
                self.rgripper.set_gripper_width(0.0)
                self.rgripper.set_state([-0.2, 0.5, -0.01])
                self.lgripper.set_state([0, 0, 0.5])
                self.rgripper.set_pose([-1.54, 0.8, -1.57])
                time.sleep(3)
                for i in range(100):
                    pb.stepSimulation()
                    width, height, rgbImg, depthImg, segImg = pb.getCameraImage(
                            360,
                            240,
                            viewMatrix=self.viewMatrix)
                    self.frames.append(rgbImg)
                    time.sleep(0.005)

                # Picking up
                self.rgripper.set_state([0.0, -0.5, 0.0])
                self.lgripper.set_state([0.0, -0.5, 0.1])

                contact_len = 0
                for i in range(100):
                    pb.stepSimulation()
                    width, height, rgbImg, depthImg, segImg = pb.getCameraImage(
                            width=360,
                            height=240,
                            viewMatrix=self.viewMatrix)
                    self.frames.append(rgbImg)
                    #time.sleep(0.005)
                    contact_len += len(pb.getContactPoints(bodyA=1, bodyB=2)) #Judge if plate and table collision
                    contact_len += len(pb.getContactPoints(bodyA=0, bodyB=2)) #Judge if plate and table collision
                print("contactlen", contact_len)
                if contact_len > 1:
                    print("Failed!!!")
                else:
                    print("Succeeded!!!")
                self.reset()
            save_video(self.frames, "sample.mp4")

        except KeyboardInterrupt:
            sys.exit()

if __name__ == '__main__':
    sim = Simulator()
    sim.rollout()
    self.pybullet.disconnect()

