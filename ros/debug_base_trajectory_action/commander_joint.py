import skrobot
import time

robot_model = skrobot.models.PR2()
robot_model.reset_manip_pose()
robot_model.head_tilt_joint.joint_angle(0.4)

viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(641, 480))
viewer.add(robot_model)

robot_model.fksolver = None
ri = skrobot.interfaces.ros.PR2ROSRobotInterface(robot_model)

for i in range(100):
    robot_model.reset_manip_pose()
    robot_model.torso_lift_joint.joint_angle(0.1)
    ri.angle_vector(robot_model.angle_vector(), time=1.0, time_scale=1.0)
    time.sleep(10)
    robot_model.init_pose()
    robot_model.torso_lift_joint.joint_angle(0.1)
    ri.angle_vector(robot_model.angle_vector(), time=1.0, time_scale=1.0)
    time.sleep(10)
