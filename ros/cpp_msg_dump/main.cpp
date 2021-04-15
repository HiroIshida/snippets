#include <fstream>
#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
namespace ser = ros::serialization;

nav_msgs::OccupancyGrid::ConstPtr occupancy_grid_;

void onOccupancyGrid(const nav_msgs::OccupancyGrid::ConstPtr & msg)
{
  ROS_INFO("occupancy grid recieved");
  occupancy_grid_ = msg;
  uint32_t serial_size = ros::serialization::serializationLength(*msg);
  boost::shared_array<uint8_t> buffer(new uint8_t[serial_size]);
  ser::OStream stream(buffer.get(), serial_size);
  ser::serialize(stream, *msg);
  //std::ofstream stream("tmp.out");
}

int main(int argc, char** argv){
  ros::init(argc, argv, "dumper");
  ros::NodeHandle nh;

  ros::Subscriber sub = nh.subscribe("/planning/scenario_planning/parking/costmap_generator/occupancy_grid", 1, onOccupancyGrid);
  nav_msgs::OccupancyGrid grid;
  ros::spin();
  return 0;
}
