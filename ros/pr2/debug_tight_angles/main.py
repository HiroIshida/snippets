import argparse
from skrobot.models.pr2 import PR2
from skrobot.interfaces import PR2ROSRobotInterface
from control_msgs.msg import FollowJointTrajectoryActionGoal
from sensor_msgs.msg import JointState
import time
import rospy


parser = argparse.ArgumentParser()
parser.add_argument("--test_larm", action="store_true")
parser.add_argument("--tight", action="store_true")
args = parser.parse_args()

robot_model = PR2(use_tight_joint_limit=args.tight)
robot_model.reset_manip_pose()
ri = PR2ROSRobotInterface(robot_model)

test_larm = args.test_larm

if test_larm:
    robot_model.l_shoulder_pan_joint.joint_angle(0.5065369606018066)
    robot_model.l_shoulder_lift_joint.joint_angle(-0.4600617587566376)
    robot_model.l_upper_arm_roll_joint.joint_angle(1.7546762228012085)
    robot_model.l_forearm_roll_joint.joint_angle(-3.6248462200164795)
    robot_model.l_elbow_flex_joint.joint_angle(-1.2340552806854248)
    robot_model.l_wrist_flex_joint.joint_angle(-0.6824826002120972)
    robot_model.l_wrist_roll_joint.joint_angle(-3.154172658920288)
else:
    robot_model.r_shoulder_pan_joint.joint_angle(-0.5065369606018066)
    robot_model.r_shoulder_lift_joint.joint_angle(-0.4600617587566376)
    robot_model.r_upper_arm_roll_joint.joint_angle(-1.7546762228012085)
    robot_model.r_forearm_roll_joint.joint_angle(-3.6248462200164795)
    robot_model.r_elbow_flex_joint.joint_angle(-1.2340552806854248)
    robot_model.r_wrist_flex_joint.joint_angle(-0.6824826002120972)
    robot_model.r_wrist_roll_joint.joint_angle(-3.154172658920288)

def callback_action_goal(msg: FollowJointTrajectoryActionGoal):
    print("action goal")
    for jn, action_angle in zip(msg.goal.trajectory.joint_names, msg.goal.trajectory.points[1].positions):
        ref_angle = robot_model.__dict__[jn].joint_angle()
        rospy.loginfo("{}: ref {} -> action {}".format(jn, ref_angle, action_angle))
if test_larm:
    rospy.Subscriber("/l_arm_controller/follow_joint_trajectory/goal", FollowJointTrajectoryActionGoal, callback=callback_action_goal)
else:
    rospy.Subscriber("/r_arm_controller/follow_joint_trajectory/goal", FollowJointTrajectoryActionGoal, callback=callback_action_goal)

ri.angle_vector(robot_model.angle_vector())
ri.wait_interpolation()
time.sleep(3)

def callback_joint_state(msg: JointState):
    print("joint state")
    for jn, state_angle in zip(msg.name, msg.position):
        filter_prefix = "l_" if test_larm else "r_"
        if jn.startswith(filter_prefix) and not "gripper" in jn:
            ref_angle = robot_model.__dict__[jn].joint_angle()
            error = abs(ref_angle - state_angle)
            if error > 0.01:
                rospy.logerr("{}: ref {} -> state {}".format(jn, ref_angle, state_angle))
            else:
                rospy.loginfo("{}: ref {} -> state {}".format(jn, ref_angle, state_angle))

    # kill node
    rospy.signal_shutdown("kill node")

rospy.Subscriber("/joint_states", JointState, callback=callback_joint_state)
