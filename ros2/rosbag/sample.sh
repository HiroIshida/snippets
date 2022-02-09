#ros2 bag record -o tmp /vehicle/* /ishida/* /planning/* /map/* /tf
ros2 bag record -o rosbag-$(date "+%Y%m%d-%H%M%S") --regex "/planning/.*|vehicle/.*|/planning/.*|/map/.*|/tf.*|/ishida.*|/robot_description"
