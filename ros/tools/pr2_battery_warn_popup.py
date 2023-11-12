#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32
import tkinter as tk
from threading import Thread

def show_warning(battery_index, value):
    root = tk.Tk()
    root.title("Battery Warning")
    message = "Warning: Battery {} is low: {}%".format(battery_index, value)
    label = tk.Label(root, text=message, height=10, width=50, font=('Times', 20))
    label.pack()
    root.after(2000, root.destroy)
    root.mainloop()

def battery_callback(data, args):
    battery_index = args[0]
    if data.data < 50.0:
        rospy.logwarn("Battery {} is low: {}%".format(battery_index, data.data))
        warning_thread = Thread(target=show_warning, args=(battery_index, data.data))
        warning_thread.start()

def battery_monitor():
    rospy.init_node('battery_monitor', anonymous=True)

    # Subscribe to all battery topics
    rospy.Subscriber("/visualization/battery/value0", Float32, battery_callback, (0,))
    rospy.Subscriber("/visualization/battery/value1", Float32, battery_callback, (1,))
    rospy.Subscriber("/visualization/battery/value2", Float32, battery_callback, (2,))
    rospy.Subscriber("/visualization/battery/value3", Float32, battery_callback, (3,))

    rospy.spin()

if __name__ == '__main__':
    try:
        battery_monitor()
    except rospy.ROSInterruptException:
        pass
