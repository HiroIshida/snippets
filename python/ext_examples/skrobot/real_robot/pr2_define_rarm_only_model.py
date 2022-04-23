from skrobot.models import PR2

class PR2RarmOnly(PR2):
    def default_controller(self):
        return [
                self.rarm_controller,
                self.head_controller,
                self.torso_controller]

class PR2LarmOnly(PR2):
    def default_controller(self):
        return [
                self.larm_controller,
                self.head_controller,
                self.torso_controller]

model = PR2_rarm_only()
robot_model.reset_manip_pose()
robot_model.reset_pose()

ri = skrobot.interfaces.ros.PR2ROSRobotInterface(robot_model)
ri.angle_vector(robot_model.angle_vector(), time=1.0, time_scale=1.0)


