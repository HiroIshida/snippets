#include <ros/ros.h>
#include <ros/service_client.h>
#include <gtest/gtest.h>
#include <simple_rostest/AddTwoInts.h>

std::shared_ptr<ros::NodeHandle> nh;

TEST(TESTSuite, addTwoInts)
{
  ros::ServiceClient client = nh->serviceClient<simple_rostest::AddTwoInts>(
      "add_two_ints");
  bool exists(client.waitForExistence(ros::Duration(1)));
  EXPECT_TRUE(exists);

  simple_rostest::AddTwoInts srv;
  srv.request.a = 5;
  srv.request.b = 8;
  client.call(srv);

  EXPECT_EQ(srv.response.sum, srv.request.a + srv.request.b);
}

int main(int argc,
         char **argv)
{
  ros::init(argc, argv, "add_two_ints_service_client");
  nh.reset(new ros::NodeHandle);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

