import rospy
from pr2_mechanism_msgs.srv import SwitchController
from pr2_mechanism_msgs.srv import SwitchControllerRequest
from pr2_mechanism_msgs.srv import SwitchControllerResponse

rospy.init_node("test")

all_controllers = ["r_arm_controller"]
all_loose_controllers = ["r_arm_controller_loose"]

srvname = '/pr2_controller_manager/switch_controller'
rospy.wait_for_service(srvname)
switch_controller = rospy.ServiceProxy(srvname, SwitchController)
def stop():
    resp = switch_controller(
            start_controllers=all_controllers, 
            stop_controllers=all_loose_controllers,
            strictness=2
            )
    print(resp)

def start():
    resp = switch_controller(
            start_controllers=all_loose_controllers, 
            stop_controllers=all_controllers,
            strictness=2
            )
    print(resp)

start()
