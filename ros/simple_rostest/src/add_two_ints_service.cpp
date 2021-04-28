#include <ros/ros.h>
#include <simple_rostest/AddTwoInts.h>

bool AddTwoInts(simple_rostest::AddTwoIntsRequest &req,
                simple_rostest::AddTwoIntsResponse &res)
{
  res.sum = req.a + req.b;
  return true;
}

int main(int argc,
         char** argv)
{
  ros::init(argc, argv, "add_two_ints_srv");
  ros::NodeHandle nh;

  ros::AsyncSpinner spinner(1);
  spinner.start();

  // Advertise service
  ros::ServiceServer service = nh.advertiseService("add_two_ints", AddTwoInts);

  ros::waitForShutdown();
  spinner.stop();
  return 0;
}

