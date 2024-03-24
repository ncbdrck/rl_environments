#!/bin/python3
"""
This script is used to test the kinect sensor in the simulation environment.
- find the resolution of the image
- find the resolution of the depth image
- find camera calibration parameters

Usage:
1. launch simulation environment with kinect sensor
2. run the script
3. check the resolution of the image and depth image
"""
import cv2

import rospy
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import PointCloud2, Image, CameraInfo


class KinectTest:
    def __init__(self, namespace='/kinect'):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(namespace + '/rgb/image_raw', Image, self.image_callback)
        self.depth_sub = rospy.Subscriber(namespace + '/depth/image_raw', Image, self.depth_callback)

        # get camera calibration parameters
        self.camera_info = rospy.wait_for_message(namespace + '/rgb/camera_info', CameraInfo)
        print('Camera calibration parameters')
        print('width:', self.camera_info.width)
        print('height:', self.camera_info.height)
        print('K:', np.array(self.camera_info.K).reshape(3, 3))
        print('D:', np.array(self.camera_info.D))
        print('R:', np.array(self.camera_info.R).reshape(3, 3))
        print('P:', np.array(self.camera_info.P).reshape(3, 4))


    def image_callback(self, data):
        image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        print("Shape of the image bgr8:", image.shape)

        # convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print("Shape of the image RGB:", image.shape)

        # show the image
        # cv2.imshow('image', image)
        # cv2.waitKey(1)

    def depth_callback(self, data):
        depth = self.bridge.imgmsg_to_cv2(data, '32FC1')
        print("Shape of depth:", depth.shape)

        # show the depth image
        cv2.imshow('depth', depth)
        cv2.waitKey(1)


if __name__ == '__main__':
    rospy.init_node('kinect_test')
    kinect = KinectTest(namespace='/head_mount_kinect2')
    rospy.spin()