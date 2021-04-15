#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>

nav_msgs::OccupancyGrid::ConstPtr occupancy_grid_;

void onOccupancyGrid(const nav_msgs::OccupancyGrid::ConstPtr & msg)
{
  ROS_INFO("occupancy grid recieved");
  occupancy_grid_ = msg;
}

int main(int argc, char** argv){
  ros::init(argc, argv, "dumper");
  ros::NodeHandle nh;

  ros::Subscriber sub = nh.subscribe("input/occupancy_grid", 1, onOccupancyGrid);
  nav_msgs::OccupancyGrid grid;
  ros::spin();
  return 0;
}
