from euslisp_command_srvs.srv import EuslispDirectCommand
import rospy

print("waiting...")
rospy.wait_for_service("command")
print("connected.")
proxy = rospy.ServiceProxy("command", EuslispDirectCommand)

#proxy("""(load "package://pr2eus/pr2-interface.l")""")
#proxy("""(pr2-init)""")
#proxy("""(print (send *ri* :state :potentio-vector))""")
#proxy("""(send *ri* :speak-jp "てすと")""")
proxy("""
        (progn 
            (load "package://pr2eus/pr2-interface.l")
            (pr2-init)
            (send *ri* :speak-jp "てすと"))""")


