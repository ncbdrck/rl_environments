#!/bin/python3
"""
This script is used to test Robot Environments
"""
import rospy
import multiros
from multiros.utils import ros_common, gazebo_core
import gymnasium as gym
import uniros as uni_gym

from rl_environments.rx200.sim.robot_envs import rx200_robot_sim_zed2, rx200_robot_sim

if __name__ == '__main__':
    # Kill all processes related to previous runs
    ros_common.kill_all_ros_and_gazebo()

    # launch gazebo
    gazebo_core.launch_gazebo(launch_roscore=False, paused=False, pub_clock_frequency=100, gui=True)

    # launch robot environment
    rospy.init_node('test_MyRobotGoalEnv')
    gym.make("RX200RobotEnv-v0")