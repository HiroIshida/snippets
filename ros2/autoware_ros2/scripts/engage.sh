#/bin/bash
ros2 topic pub --once /autoware/engage autoware_vehicle_msgs/msg/Engage "engage: true"
