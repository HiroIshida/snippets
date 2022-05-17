import math
import skrobot
from skrobot.model import Joint
from skrobot.interfaces.ros import PR2ROSRobotInterface  # type: ignore

robot_model = skrobot.models.PR2()
ri = PR2ROSRobotInterface(robot_model)

def init():
    print("init pose")
    robot_model.init_pose()
    ri.angle_vector(robot_model.angle_vector(), time=2.0, time_scale=1.0)
    ri.wait_interpolation()

init()

print("reset pose by angle-vector-sequence")
robot_model.reset_pose()
av_target = robot_model.angle_vector()
av_current = ri.potentio_vector()
n_split = 10
av_seq = [av_current + i * (av_target - av_current) / (n_split - 1) for i in range(n_split)]
ri.angle_vector_sequence(av_seq)
ri.wait_interpolation()
