#!/usr/bin/env python

import rospy
from sensor_msgs.msg import CameraInfo
from sensor_msgs.srv import SetCameraInfo, SetCameraInfoRequest

"""
Original CameraInfo message: (by tsukamoto)

header: 
  seq: 139502
  stamp: 
    secs: 1701675336
    nsecs: 761567696
  frame_id: "head_mount_kinect_rgb_optical_frame"
height: 480
width: 640
distortion_model: "plumb_bob"
D: [0.0, 0.0, 0.0, 0.0, 0.0]
K: [525.0, 0.0, 319.5, 0.0, 525.0, 239.5, 0.0, 0.0, 1.0]
R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
P: [525.0, 0.0, 319.5, 0.0, 0.0, 525.0, 239.5, 0.0, 0.0, 0.0, 1.0, 0.0]
binning_x: 0
binning_y: 0
roi: 
  x_offset: 0
  y_offset: 0
  height: 0
  width: 0
  do_rectify: False
"""

def set_camera_info():
    rospy.init_node('set_kinect_camera_info', anonymous=True)
    rospy.wait_for_service('/kinect_head/rgb/set_camera_info')

    try:
        set_camera_info_service = rospy.ServiceProxy('/kinect_head/rgb/set_camera_info', SetCameraInfo)

        camera_info = CameraInfo()
        camera_info.header.stamp = rospy.Time.now()
        camera_info.header.frame_id = "head_mount_kinect_rgb_optical_frame"
        camera_info.height = 480
        camera_info.width = 640
        camera_info.distortion_model = "plumb_bob"
        camera_info.D = [0.0, 0.0, 0.0, 0.0, 0.0]
        camera_info.K = [525.0, 0.0, 319.5, 0.0, 525.0, 239.5, 0.0, 0.0, 1.0]
        camera_info.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        camera_info.P = [525.0, 0.0, 319.5, 0.0, 0.0, 525.0, 239.5, 0.0, 0.0, 0.0, 1.0, 0.0]
        camera_info.binning_x = 0
        camera_info.binning_y = 0
        camera_info.roi.x_offset = 0
        camera_info.roi.y_offset = 0
        camera_info.roi.height = 0
        camera_info.roi.width = 0
        camera_info.roi.do_rectify = False

        # Create a SetCameraInfoRequest message
        request = SetCameraInfoRequest()
        request.camera_info = camera_info

        # Call the service
        response = set_camera_info_service(request)

        return response
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)

if __name__ == '__main__':
    result = set_camera_info()
    print("SetCameraInfo response: ", result)
