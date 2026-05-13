#!/bin/python3
"""
This script is used to test Robot Environments
"""
import rospy
import multiros
from multiros.utils import gazebo_core
import gymnasium as gym
import uniros as uni_gym

from rl_environments.rx200.sim.robot_envs import rx200_robot_sim_zed2, rx200_robot_sim

if __name__ == '__main__':
    # Note: a previous version of this script called
    # ros_common.kill_all_ros_and_gazebo() here, which issues a
    # host-wide ``killall -9`` against rosmaster/roslaunch/gzserver/etc.
    # That would clobber any other ROS or Gazebo session running on the
    # machine, including teammates' work. If you need to clean up
    # leftover instances from a previous run, do so explicitly from
    # your shell before launching this script — the API is now
    # ros_common.kill_all_host_ros_and_gazebo() and the host-wide
    # scope is made explicit by the name.

    # launch gazebo
    gazebo_core.launch_gazebo(launch_roscore=False, paused=False, pub_clock_frequency=100, gui=True)

    # launch robot environment
    rospy.init_node('test_MyRobotGoalEnv')
    gym.make("RX200RobotEnv-v0")