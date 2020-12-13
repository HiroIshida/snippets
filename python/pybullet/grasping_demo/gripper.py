import utils

import pybullet
import pybullet as pb


class Gripper(object):
    def __init__(self, urdf_path=None):

        def construct_tables(robot):
            # construct tables
            # table (joint_name -> id) (link_name -> id)
            link_table = {pb.getBodyInfo(robot)[
                0].decode('UTF-8'): -1}
            joint_table = {}
            def heck(path): return "_".join(path.split("/"))
            for _id in range(pb.getNumJoints(robot)):
                joint_info = pb.getJointInfo(
                    robot, _id)
                joint_id = joint_info[0]

                joint_name = joint_info[1].decode('UTF-8')
                joint_table[joint_name] = joint_id
                name_ = joint_info[12].decode('UTF-8')
                name = heck(name_)
                link_table[name] = _id
            return link_table, joint_table

        if urdf_path is None:
            urdf_path = self.urdf_path

        self.gripper = pb.loadURDF(urdf_path, useFixedBase=True)
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
            for i in [8, 9, 10, 11]:
                pb.resetJointState(self.gripper, i, angle,
                                   targetVelocity=0.0)
        else:
            for i in [8, 9, 10, 11]:
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


class LGripper(Gripper):
    urdf_path = "pr2_description/urdf/pr2_lgripper.urdf"


class RGripper(Gripper):
    urdf_path = "pr2_description/urdf/pr2_rgripper.urdf"
