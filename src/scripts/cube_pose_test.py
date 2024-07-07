#!/bin/python3
"""
This script is used to test the pose of a cube in Gazebo.
- The cube is spawned in Gazebo
- The pose of the cube is retrieved
- The cube is removed from Gazebo

"""
import cv2

import rospy
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from tf.transformations import euler_from_matrix

# core modules of the multiros framework
from multiros.utils import gazebo_core
from multiros.utils import gazebo_models
from multiros.utils import ros_common

class CubePoseTest:
    def __init__(self, load_table=True, real_time=True):
        self.load_table = load_table
        self.real_time = real_time

    def spawn_cube_in_gazebo(self, model_pos_x, model_pos_y):
        """
        Spawn a cube in Gazebo

        Args:
            model_pos_x: x-coordinate of the cube
            model_pos_y: y-coordinate of the cube

        Returns:
            done: True if the cube is spawned successfully
        """
        if self.load_table:
            model_pos_z = 0.795
        else:
            model_pos_z = 0.015

        # spawn a cube
        done = gazebo_models.spawn_sdf_model_gazebo(pkg_name="reactorx200_description", file_name="block.sdf",
                                                    model_folder="/models/block",
                                                    model_name="red_cube", namespace="/rx200",
                                                    pos_x=model_pos_x,
                                                    pos_y=model_pos_y,
                                                    pos_z=model_pos_z)

        # above function pauses the simulation, so we need to unpause it
        if self.real_time:
            gazebo_core.unpause_gazebo()

        return done

    def get_model_pose(self, model_name="red_cube", rpy=True):
        """
        Get the pose of an object in Gazebo

        Args:
            model_name: name of the object whose pose is to be retrieved
            rpy: True if the orientation is to be returned as euler angles (default: True)

        Returns:
            success: True if the pose is retrieved successful
            position: position of the object as a numpy array
            orientation: orientation of the object as a numpy array (rpy or quaternion)
        """

        if not self.real_time:
            gazebo_core.unpause_gazebo()

        # pose is a geometry_msgs/Pose Message
        header, pose, twist, success = gazebo_models.gazebo_get_model_state(model_name=model_name,
                                                                            relative_entity_name="rx200/base_link")

        if not self.real_time:
            gazebo_core.pause_gazebo()

        if success:
            if rpy:
                orientation = euler_from_matrix([[pose.orientation.x, pose.orientation.y, pose.orientation.z],
                                                 [pose.orientation.y, pose.orientation.w, pose.orientation.z],
                                                 [pose.orientation.z, pose.orientation.y, pose.orientation.w]])

                orientation = np.array(orientation, dtype=np.float32)
            else:
                orientation = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z,
                                        pose.orientation.w], dtype=np.float32)

            position = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float32)

            # return position and orientation as numpy arrays
            return success, position, orientation

        # if the pose is not retrieved successfully
        return success, None, None

    def remove_cube_in_gazebo(self):
        """
        Remove the cube from Gazebo
        """
        done = gazebo_models.remove_model_gazebo(model_name="red_cube")

        # above function pauses the simulation, so we need to unpause it
        if self.real_time:
            gazebo_core.unpause_gazebo()

        return done


if __name__ == '__main__':
    rospy.init_node('kinect_test')
    CubeObject = CubePoseTest(load_table=True, real_time=True)

    # get the pose of the cube
    success, position, orientation = CubeObject.get_model_pose(model_name="red_cube", rpy=True)
    if success:
        print("Cube Pose:")
        print("Position: ", position)
        print("Orientation: ", orientation)
    else:
        print("Cube not found in Gazebo")

    # remove the cube from Gazebo
    CubeObject.remove_cube_in_gazebo()

    # spawn the cube in Gazebo
    done = CubeObject.spawn_cube_in_gazebo(model_pos_x=0.180, model_pos_y=0.0)

    # get the pose of the cube
    success, position, orientation = CubeObject.get_model_pose(model_name="red_cube", rpy=True)
    if success:
        print("Cube Pose:")
        print("Position: ", position)
        print("Orientation: ", orientation)
    else:
        print("Cube not found in Gazebo")

