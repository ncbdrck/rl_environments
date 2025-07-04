#!/bin/python3

from typing import Any, Optional, Dict

import rospy
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register
import scipy.spatial

# Custom robot env
from rl_environments.ned2.real.robot_envs import ned2_robot_goal_real

# core modules of the framework
from realros.utils import ros_common
from realros.utils import ros_markers

# Register your environment using the OpenAI register method to utilize gym.make("MyTaskGoalEnv-v0").
# register(
#     id='NED2ReacherGoalReal-v0',
#     entry_point='rl_environments.ned2.real.task_envs.reach.ned2_reach_goal_real:NED2ReacherGoalEnv',
#     max_episode_steps=100,
# )

"""
This is the v0 of the NED2 Reacher Goal conditioned Task Environment.
- option to use vision sensors - depth and rgb images
- action space is joint positions of the robot arm or xyz position of the end effector. No gripper control
- reward is sparse or dense
- goal is to reach a goal position
- Only works in real-time mode no sequential mode
- uses kinect v2 or ZED2 for vision
"""


class RX200ReacherGoalEnv(ned2_robot_goal_real.NED2RobotGoalEnv):
    """
    This Task env is for a simple Reach Task with the NED2 robot.

    The task is done if
        * The robot reached the goal

    Here
        * Action Space - Continuous (5 actions for joints or 3 xyz position of the end effector)
        * Observation - Continuous (simp obs or rgb/depth image or a combination)
        * Desired Goal - Goal we are trying to reach
        * Achieved Goal - Position of the EE

    Init Args:
        * new_roscore: Whether to launch a new roscore or not. If False, it is assumed that a roscore is already running.
        * roscore_port: Port of the roscore to be launched. If None, a random port is chosen.
        * seed: Seed for the random number generator.
        * close_env_prompt: Whether to prompt the user to close the env or not.
        * reward_type: Type of reward to be used. It Can be "Sparse" or "Dense".
        * delta_action: Whether to use the delta actions or the absolute actions.
        * delta_coeff: Coefficient to be used for the delta actions.
        * ee_action_type: Whether to use the end effector action space or the joint action space.
        * environment_loop_rate: Rate at which the environment loop should run. (in Hz)
        * action_cycle_time: Time to wait between two consecutive actions. (in seconds)
        * use_smoothing: Whether to use smoothing for actions or not.
        * default_port: Whether to use the default port for the roscore or not.
        * rgb_obs_only: Whether to use only the RGB image as the observations or not.
        * normal_obs_only: Whether to use only the traditional observations or not.
        * rgb_plus_normal_obs: Whether to use RGB image and traditional observations or not.
        * rgb_plus_depth_plus_normal_obs: Whether to use RGB image, Depth image and traditional observations or not.
        * debug: Whether to print debug messages or not.
        * action_speed: set the speed to complete the trajectory. default in 0.5 seconds
        * simple_dense_reward: Whether to use a simple dense reward or not.
        * log_internal_state: Whether to log the internal state of the environment or not.
        * use_kinect: Whether to use the kinect sensor or not.
        * use_zed2: Whether to use the ZED2 camera or not.
        * remote_ip: IP address of the remote machine where the ROS master is running.
        * local_ip: IP address of the local machine where the ROS node is running.
        * multi_device_mode: Whether to use multi-device mode or not. If True, remote_ip and local_ip must be provided.
    """

    def __init__(self, new_roscore: bool = False, roscore_port: str = None, seed: int = None,
                 close_env_prompt: bool = True, reward_type: str = "sparse",
                 delta_action: bool = True, delta_coeff: float = 0.05,
                 environment_loop_rate: float = None, action_cycle_time: float = 0.0,
                 use_smoothing: bool = False, default_port=True, ee_action_type: bool = False,
                 rgb_obs_only: bool = False, normal_obs_only: bool = True, rgb_plus_normal_obs: bool = False,
                 rgb_plus_depth_plus_normal_obs: bool = False, debug: bool = False, action_speed: float = 0.5,
                 simple_dense_reward: bool = False, log_internal_state: bool = False, use_kinect: bool = False,
                 use_zed2: bool = False, remote_ip: str = None, local_ip:str = None, multi_device_mode: bool = False,
                 ):


        """
        variables to keep track of ros port
        """
        ros_port = roscore_port

        """
        Initialise the env
        """
        if multi_device_mode:
            if remote_ip is not None and local_ip is not None and ros_port is not None:
                ros_common.change_ros_master_multi_device(remote_ip=remote_ip,
                                                      local_ip=local_ip, remote_ros_port=ros_port)
            else:
                rospy.logerr("Remote IP and Local IP must be provided for multi-device mode.")

        # launch a new roscore with default port
        elif default_port:
            ros_port = self._launch_roscore(default_port=default_port)

        # Launch new roscore
        elif new_roscore:
            ros_port = self._launch_roscore(port=roscore_port)

        # ros_port of the already running roscore
        elif roscore_port is not None:
            ros_port = roscore_port

            # change to new rosmaster
            ros_common.change_ros_master(ros_port)

        else:
            """
            Check for roscore
            """
            if ros_common.is_roscore_running() is False:
                print("roscore is not running! Launching a new roscore!")
                ros_port = self._launch_roscore(port=roscore_port)

        # init the ros node
        if ros_port is not None:
            self.node_name = "NED2ReacherGoalEnvReal" + "_" + ros_port
        else:
            self.node_name = "NED2ReacherGoalEnvReal"

        rospy.init_node(self.node_name, anonymous=True)

        """
        Provide a description of the task.
        """
        rospy.loginfo(f"Starting {self.node_name}")

        """
        Exit the program if
        - (1/environment_loop_rate) is greater than action_cycle_time
        """
        if (1.0 / environment_loop_rate) > action_cycle_time:
            rospy.logerr("The environment loop rate is greater than the action cycle time. Exiting the program!")
            rospy.signal_shutdown("Exiting the program!")
            exit()

        """
        log internal state - using rospy loginfo, logwarn, logerr
        """
        self.log_internal_state = log_internal_state

        """
        Reward Architecture
            * Dense - Default
            * Sparse - -1 if not done and 1 if done
            * All the others and misspellings - default to "Dense" reward
        """
        if reward_type.lower() == "sparse":
            self.reward_arc = "Sparse"

        elif reward_type.lower() == "dense":
            self.reward_arc = "Dense"

        else:
            rospy.logwarn(f"The given reward architecture '{reward_type}' not found. Defaulting to Dense!")
            self.reward_arc = "Dense"

        # check if we are using simple dense reward
        # for simple, we only return the distance to the goal (negative Euclidean distance = -d)
        self.simple_dense_reward = simple_dense_reward

        """
        Use action as deltas
        """
        self.delta_action = delta_action
        self.delta_coeff = delta_coeff

        """
        Use smoothing for actions
        """
        self.use_smoothing = use_smoothing
        self.action_cycle_time = action_cycle_time

        """
        Action speed - Time to complete the trajectory
        """
        self.action_speed = action_speed

        """
        Observation space
            * RGB image only
            * traditional observations only (default)
            * RGB image and traditional observations (combined)
            * RGB image, Depth image and traditional observations (combined)
        """
        self.rgb_obs = rgb_obs_only
        self.normal_obs = normal_obs_only
        self.rgb_plus_normal_obs = rgb_plus_normal_obs
        self.rgb_plus_depth_plus_normal_obs = rgb_plus_depth_plus_normal_obs

        """
        Action space
            * Joint action space (default)
            * End effector action space
        """
        self.ee_action_type = ee_action_type

        """
        Debug
        """
        self.debug = debug

        """
        Load YAML param file
        """

        # add to ros parameter server
        ros_common.ros_load_yaml(pkg_name="rl_environments", file_name="ned2_reach_task_config.yaml", ns="/")
        self._get_params()

        """
        Define the action space.
        """
        # Joint action space or End effector action space
        # ROS and Gazebo often use double-precision (64-bit),
        # but we are using single-precision (32-bit) as it is typical for RL implementations.

        if self.ee_action_type:
            # EE action space
            self.max_ee_values = np.array([self.position_ee_max["x"], self.position_ee_max["y"],
                                           self.position_ee_max["z"]])
            self.min_ee_values = np.array([self.position_ee_min["x"], self.position_ee_min["y"],
                                           self.position_ee_min["z"]])

            self.action_space = spaces.Box(low=np.array(self.min_ee_values), high=np.array(self.max_ee_values),
                                           dtype=np.float32)

        else:
            # Joint action space
            self.action_space = spaces.Box(low=np.array(self.min_joint_values), high=np.array(self.max_joint_values),
                                           dtype=np.float32)

        """
        Define the observation space.

        # observation
        01. EE pos - 3
        02. Vector to the goal (normalized linear distance) - 3
        03. Euclidian distance (ee to reach goal)- 1
        04. Current Joint values - 6  # this is 8 if we have the gripper
        05. Previous action - 6
        06. Joint velocities - 6

        total: (3x2) + 1 + (6 or 3) + (6x2) 

        # depth image
        480x640 32FC1

        # rgb image
        480x640X3 RGB images

        So observation space is a dictionary with
            observation: Box(25 or 22) or Dict(25 or 22 and 480x640x3 or 480x640)
            achieved_goal: EE pos - 3 elements
            desired_goal: Goal pos - 3 elements
        """

        # ---- ee pos
        observations_high_ee_pos_range = np.array(
            np.array([self.position_ee_max["x"], self.position_ee_max["y"], self.position_ee_max["z"]]))
        observations_low_ee_pos_range = np.array(
            np.array([self.position_ee_min["x"], self.position_ee_min["y"], self.position_ee_min["z"]]))

        # ---- vector to the goal - normalized linear distance
        observations_high_vec_ee_goal = np.array([1.0, 1.0, 1.0])
        observations_low_vec_ee_goal = np.array([-1.0, -1.0, -1.0])

        # ---- Euclidian distance
        observations_high_dist = np.array([self.max_distance])
        observations_low_dist = np.array([0.0])

        # ---- joint values
        observations_high_joint_values = self.max_joint_angles.copy()
        observations_low_joint_values = self.min_joint_angles.copy()

        # ---- previous action
        if self.ee_action_type:
            observations_high_prev_action = self.max_ee_values.copy()
            observations_low_prev_action = self.min_ee_values.copy()
        else:
            observations_high_prev_action = self.max_joint_values.copy()
            observations_low_prev_action = self.min_joint_values.copy()

        # ---- joint velocities
        observations_high_joint_vel = self.max_joint_vel.copy()
        observations_low_joint_vel = self.min_joint_vel.copy()

        high = np.concatenate(
            [observations_high_ee_pos_range, observations_high_vec_ee_goal, observations_high_dist,
             observations_high_joint_values, observations_high_prev_action, observations_high_joint_vel, ])

        low = np.concatenate(
            [observations_low_ee_pos_range, observations_low_vec_ee_goal, observations_low_dist,
             observations_low_joint_values, observations_low_prev_action, observations_low_joint_vel, ])

        # Define the traditional observation space
        self.observations = spaces.Box(low=low, high=high, dtype=np.float32)

        # Define the depth image space (480x640 32FC1) - this uses 32-bit float
        self.depth_image_space = spaces.Box(low=0, high=1, shape=(480, 640), dtype=np.float32)

        # Define the image space (480x640X3 RGB images) - this uses 8-bit unsigned int
        self.rgb_image_space = spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8)

        """
        Achieved goal (EE pose) - 3
        """
        high_achieved_goal_pos_range = np.array(
            np.array([self.position_achieved_goal_max["x"], self.position_achieved_goal_max["y"],
                      self.position_achieved_goal_max["z"]]))
        low_achieved_goal_pos_range = np.array(
            np.array([self.position_achieved_goal_min["x"], self.position_achieved_goal_min["y"],
                      self.position_achieved_goal_min["z"]]))

        self.achieved_goal_space = spaces.Box(low=low_achieved_goal_pos_range, high=high_achieved_goal_pos_range,
                                              dtype=np.float32)

        """
        Desired goal (Goal pose) - 3
        """
        high_desired_goal_pos_range = np.array(
            np.array([self.position_desired_goal_max["x"], self.position_desired_goal_max["y"],
                      self.position_desired_goal_max["z"]]))
        low_desired_goal_pos_range = np.array(
            np.array([self.position_desired_goal_min["x"], self.position_desired_goal_min["y"],
                      self.position_desired_goal_min["z"]]))

        self.desired_goal_space = spaces.Box(low=low_desired_goal_pos_range, high=high_desired_goal_pos_range,
                                             dtype=np.float32)

        """
        Define the overall observation space
        """
        if self.normal_obs:
            use_kinect = False  # to pass to the superclass
            use_zed2 = False  # to pass to the superclass
            self.observation_space = spaces.Dict({
                'observation': self.observations,
                'achieved_goal': self.achieved_goal_space,
                'desired_goal': self.desired_goal_space
            })

        elif self.rgb_obs:
            self.observation_space = spaces.Dict({
                'observation': self.rgb_image_space,
                'achieved_goal': self.achieved_goal_space,
                'desired_goal': self.desired_goal_space
            })

        elif self.rgb_plus_normal_obs:
            # Define a combined observation space
            obs = spaces.Dict({
                "rgb_image": self.rgb_image_space,
                "observations": self.observations
            })

            # Define the overall observation space
            self.observation_space = spaces.Dict({
                'observation': obs,
                'achieved_goal': self.achieved_goal_space,
                'desired_goal': self.desired_goal_space
            })

        elif self.rgb_plus_depth_plus_normal_obs:
            # Define a combined observation space
            obs = spaces.Dict({
                "depth_image": self.depth_image_space,
                "rgb_image": self.rgb_image_space,
                "observations": self.observations
            })

            # Define the overall observation space
            self.observation_space = spaces.Dict({
                'observation': obs,
                'achieved_goal': self.achieved_goal_space,
                'desired_goal': self.desired_goal_space
            })

        # if none of the above, use the traditional observation space
        else:
            use_kinect = False
            use_zed2 = False
            self.observation_space = spaces.Dict({
                'observation': self.observations,
                'achieved_goal': self.achieved_goal_space,
                'desired_goal': self.desired_goal_space
            })

        """
        Goal space for sampling
        """
        high_goal_pos_range = np.array(
            np.array([self.position_goal_max["x"], self.position_goal_max["y"], self.position_goal_max["z"]]))
        low_goal_pos_range = np.array(
            np.array([self.position_goal_min["x"], self.position_goal_min["y"], self.position_goal_min["z"]]))

        # -- goal space for sampling
        self.goal_space = spaces.Box(low=low_goal_pos_range, high=high_goal_pos_range, dtype=np.float32,
                                     seed=seed)

        """
        Workspace so we can check if the action is within the workspace
        """
        # ---- Workspace
        high_workspace_range = np.array(
            np.array([self.workspace_max["x"], self.workspace_max["y"], self.workspace_max["z"]]))
        low_workspace_range = np.array(
            np.array([self.workspace_min["x"], self.workspace_min["y"], self.workspace_min["z"]]))

        # -- workspace space for checking
        # we don't need to set the seed here since we're not sampling from this space
        self.workspace_space = spaces.Box(low=low_workspace_range, high=high_workspace_range, dtype=np.float32)

        """
        Define subscribers/publishers and Markers as needed.
        """
        self.goal_marker = ros_markers.RosMarker(frame_id="world", ns="goal", marker_type=2, marker_topic="goal_pos",
                                                 lifetime=30.0)

        """
        Init super class.
        """
        super().__init__(ros_port=ros_port, seed=seed, close_env_prompt=close_env_prompt, use_kinect=use_kinect,
                         use_zed2=use_zed2, action_cycle_time=action_cycle_time,
                         remote_ip=remote_ip, local_ip=local_ip, multi_device_mode=multi_device_mode)

        # for smoothing
        if self.use_smoothing:
            if self.ee_action_type:
                self.action_vector = np.zeros(3, dtype=np.float32)
            else:
                self.action_vector = np.zeros(6, dtype=np.float32)

        # we can use this to set a time for ros_controllers to complete the action
        self.environment_loop_time = 1.0 / environment_loop_rate  # in seconds

        self.prev_action = None  # we need this for observation

        if environment_loop_rate is not None:
            self.obs_r = None
            self.reward_r = None
            self.terminated_r = False
            self.truncated_r = False
            self.info_r = {}
            self.current_action = None
            self.init_done = False  # we don't need to execute the loop until we reset the env

            # Debug
            if self.debug:
                self.loop_counter = 0
                self.action_counter = 0

            # create a timer to run the environment loop
            rospy.Timer(rospy.Duration(1.0 / environment_loop_rate), self.environment_loop)

        # for dense reward calculation - for more complex reward calculations
        self.action_not_in_limits = False
        self.lowest_z = self.position_goal_min["z"]
        self.movement_result = False
        self.within_goal_space = False

        """
        Finished __init__ method
        """
        rospy.loginfo(f"Finished Init of {self.node_name}")

    # -------------------------------------------------------
    #   Methods for interacting with the environment

    def _set_init_params(self, options: Optional[Dict[str, Any]] = None):
        """
        Set initial parameters for the environment.

        Here we
            1. Move the Robot to Home position
            2. Find a valid random reach goal
        """
        if self.log_internal_state:
            rospy.loginfo("Initialising the init params!")

        # Initial robot pose - Home
        self.init_pos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # make the current action None to stop execution for real time envs and also stop the env loop
        self.init_done = False  # we don't need to execute the loop until we reset the env
        self.current_action = None

        # for smoothing
        if self.use_smoothing:
            if self.ee_action_type:
                self.action_vector = np.zeros(3, dtype=np.float32)
            else:
                self.action_vector = np.zeros(6, dtype=np.float32)

        # move the robot to the home pose
        # we need to wait for the movement to finish
        # we define the movement result here so that we can use it in the environment loop (we need it for dense reward)
        self.move_NED2_object.stop_arm()
        self.movement_result = self.move_NED2_object.set_trajectory_joints(self.init_pos)
        if not self.movement_result:
            if self.log_internal_state:
                rospy.logwarn("Homing failed!")

        #  Get a random Reach goal - np.array
        # goal_found, goal_vector = self.get_random_goal()  # this checks if the goal is reachable using moveit
        goal_found, goal_vector = self.get_random_goal_no_check()

        if goal_found:
            self.reach_goal = goal_vector
            if self.log_internal_state:
                rospy.loginfo("Reach Goal--->" + str(self.reach_goal))

        else:
            # fake Reach goal - hard code one
            self.reach_goal = np.array([0.250, 0.000, 0.015], dtype=np.float32)
            if self.log_internal_state:
                rospy.logwarn("Hard Coded Reach Goal--->" + str(self.reach_goal))

        # Publish the goal pos
        self.goal_marker.set_position(position=self.reach_goal)
        self.goal_marker.publish()

        # get initial ee pos and joint values (we need this for delta actions or when we have EE action space)
        ee_pos_tmp = self.get_ee_pose()  # Get a geometry_msgs/PoseStamped msg
        self.ee_pos = np.array([ee_pos_tmp.pose.position.x, ee_pos_tmp.pose.position.y, ee_pos_tmp.pose.position.z])
        self.ee_ori = np.array([ee_pos_tmp.pose.orientation.x, ee_pos_tmp.pose.orientation.y,
                                ee_pos_tmp.pose.orientation.z,
                                ee_pos_tmp.pose.orientation.w])  # for IK calculation - EE actions
        self.joint_values = self.get_joint_angles()

        # for dense reward calculation
        self.action_not_in_limits = False
        self.within_goal_space = True

        if self.ee_action_type:
            self.prev_action = self.ee_pos.copy()  # for observation
        else:
            self.prev_action = self.init_pos.copy()  # for observation

        # We can start the environment loop now
        if self.log_internal_state:
            rospy.loginfo("Start resetting the env loop!")

        # init the real time variables
        self.obs_r = None
        self.reward_r = None
        self.terminated_r = False
        self.truncated_r = False
        self.info_r = {}

        # debug
        if self.debug:
            self.loop_counter = 0
            self.action_counter = 0

        if self.log_internal_state:
            rospy.loginfo("Done resetting the env loop!")

        self.init_done = True
        # self.current_action = self.init_pos.copy()

        if self.log_internal_state:
            rospy.loginfo("Initialising init params done--->")

    def _set_action(self, action):
        """
        Function to apply an action to the robot.

        Args:
            action: Joint positions (numpy array) or EE position (numpy array)
        """
        # save the action for observation
        self.prev_action = action.copy()

        if self.log_internal_state:
            rospy.loginfo(f"Applying real-time action---> {action}")

        self.current_action = action.copy()

        # for debugging
        if self.debug:
            self.action_counter = 0  # reset the action counter

    def _get_observation(self):
        """
        Function to get an observation from the environment.

        Returns:
            An observation representing the current state of the environment.
        """
        obs = None
        # we cannot copy a None value
        if self.obs_r is not None:
            obs = self.obs_r.copy()

        # incase we don't have an observation yet for real time envs
        if obs is None:
            obs = self.sample_observation()

        return obs.copy()

    def _get_achieved_goal(self):
        """
        Get the achieved goal from the environment.

        Returns:
            achieved_goal: EE position
        """
        # ee_pos_tmp = self.get_ee_pose()  # Get a geometry_msgs/PoseStamped msg
        # achieved_goal = np.array(
        #     [ee_pos_tmp.pose.position.x, ee_pos_tmp.pose.position.y, ee_pos_tmp.pose.position.z])
        #
        # return achieved_goal.copy()

        return self.ee_pos.copy()

    def _get_desired_goal(self):
        """
        Get the desired goal from the environment.

        Returns:
            desired_goal: Reach Goal
        """
        return self.reach_goal.copy()

    def compute_reward(self, achieved_goal, desired_goal, info) -> float:
        """
        Compute the reward for achieving a given goal.

        Args:
            achieved_goal: EE position
            desired_goal: Reach Goal
            info (dict|list): Additional information about the environment. A list for SB3 HER case.

        Returns:
            reward (float): The reward for achieving the given goal.
        """
        is_her = False

        # check if sb3 is using this with HER
        if info is not None and isinstance(info, list):
            is_her = True

        # check if we are using this with HER - my custom implementation
        elif info is not None and isinstance(info, dict):
            is_her = info.get('is_her', False)  # this is set in the HER sampler (my custom implementation)

        # Handle multiple goals for HER
        if is_her:
            if self.log_internal_state:
                rospy.loginfo("Using HER reward calculation!")

            # debug
            # print("achieved_goal", achieved_goal)
            # # print the length of the achieved_goal
            # print("len(achieved_goal)", len(achieved_goal))
            # # print the shape of the achieved_goal
            # print("achieved_goal.shape", achieved_goal.shape)
            # print("desired_goal", desired_goal)
            # print("info", info)

            # we need to set the info to a dict with is_her key as True
            info_tmp = {"is_her": True}

            rewards = [self.calculate_reward(ag, dg, info_tmp) for ag, dg in zip(achieved_goal, desired_goal)]
            return np.array(rewards, dtype=np.float32)

        reward = None
        if self.reward_r is not None and not is_her:
            reward = self.reward_r

        # incase we don't have a reward yet for real time envs
        # also for Hindsight Experience Replay
        if reward is None:
            reward = self.calculate_reward(achieved_goal, desired_goal, info)

        # Return reward as a single-element list if info is a list (SB3 HER case)
        return reward

    def compute_terminated(self, achieved_goal, desired_goal, info):
        """
        Function to check if the episode is terminated.

        Args:
            achieved_goal: EE position
            desired_goal: Reach Goal
            info (dict): Additional information about the environment.

        Returns:
            A boolean value indicating whether the episode has ended
            (e.g. because a goal has been reached or a failure condition has been triggered)
        """
        terminated = self.terminated_r
        self.info = self.info_r  # we can use this to log the success rate in stable baselines3

        # check if self.info have the is_success key
        # double-checking here
        if "is_success" not in self.info:
            if terminated:
                self.info["is_success"] = True
            else:
                self.info["is_success"] = False

        # incase we don't have a done yet for real time envs
        # unnecessary to check since we never set the terminated to None
        if terminated is None:
            terminated = self.check_if_done()

            if terminated:
                self.info["is_success"] = True  # explicitly set the success rate
            else:
                self.info["is_success"] = False

        return terminated

    def compute_truncated(self, achieved_goal, desired_goal, info):
        """
        Function to check if the episode is truncated.

        Mainly hard coded here since we are using a wrapper that sets the max number of steps and truncates the episode.

        Args:
            achieved_goal: EE position
            desired_goal: Reach Goal
            info (dict): Additional information about the environment.

        Returns:
            A boolean value indicating whether the episode has been truncated
            (e.g. because the maximum number of steps has been reached)
        """
        truncated = self.truncated_r

        return truncated

    # -------------------------------------------------------
    #   Include any custom methods available for the MyTaskEnv class

    def environment_loop(self, event):
        """
        Function for Environment loop for real time RL envs
        """

        #  we don't need to execute the loop until we reset the env
        if self.init_done:

            if self.debug:
                if self.log_internal_state:
                    rospy.loginfo(f"Starting RL loop --->: {self.loop_counter}")
                self.loop_counter += 1

            # start with the observation, reward, done and info
            self.info_r = {}
            self.obs_r = self.sample_observation()
            self.reward_r = self.calculate_reward()
            self.terminated_r = self.check_if_done()

            # Apply the action
            # we need this if we're done with the task we can break the loop in above done check
            if self.current_action is not None:
                self.execute_action(self.current_action)

                if self.debug:
                    if self.log_internal_state:
                        rospy.loginfo(f"Executing action --->: {self.action_counter}")
                    self.action_counter += 1
            else:
                self.move_NED2_object.stop_arm()  # stop the arm if there is no action

    def execute_action(self, action):
        """
        Function to apply an action to the robot.

        This method should be implemented here to apply the given action to the robot. The action could be a
        joint position command, a velocity command, or any other type of command that can be applied to the robot.

        Args:
            action: The action to be applied to the robot.
        """
        if self.log_internal_state:
            rospy.loginfo(f"Action --->: {action}")

        # --- Set the action based on the action type
        # --- EE action
        if self.ee_action_type:

            # --- Make actions as deltas
            if self.delta_action:
                # we can use smoothing using the action_cycle_time or delta_coeff
                if self.use_smoothing:
                    if self.action_cycle_time == 0.0:
                        # first derivative of the action
                        self.action_vector = self.action_vector + (self.delta_coeff * action)

                        # clip the action vector to be within -1 and 1
                        self.action_vector = np.clip(self.action_vector, -1, 1)

                        # add the action vector to the current ee pos
                        action = self.ee_pos + (self.action_vector * self.delta_coeff)

                    else:
                        # first derivative of the action
                        self.action_vector = self.action_vector + (self.action_cycle_time * action)

                        # clip the action vector to be within -1 and 1
                        self.action_vector = np.clip(self.action_vector, -1, 1)

                        # add the action vector to the current ee pos
                        action = self.ee_pos + (self.action_vector * self.action_cycle_time)

                else:
                    action = self.ee_pos + (action * self.delta_coeff)

            # check if the action is within the joint limits (for the reward calculation)
            min_ee_values = np.array(self.min_ee_values)
            max_ee_values = np.array(self.max_ee_values)
            self.action_not_in_limits = np.any(action <= (min_ee_values + 0.0001)) or np.any(
                action >= (max_ee_values - 0.0001))

            # clip the action
            action = np.clip(action, self.min_ee_values, self.max_ee_values)

            # check if we can reach the goal and within the goal space
            # check if action is within the workspace
            if self.workspace_space.contains(action):

                # calculate IK
                IK_found, joint_positions = self.calculate_ik(target_pos=action, ee_ori=self.ee_ori)

                if IK_found:
                    # execute the trajectory - EE
                    self.movement_result = self.move_arm_joints(q_positions=joint_positions,
                                                                time_from_start=self.action_speed)
                    self.within_goal_space = True

                else:
                    if self.log_internal_state:
                        rospy.logwarn(f"The action: {action} is not reachable!")
                        rospy.logdebug(f"Set action failed for --->: {action}")
                    self.movement_result = False
                    self.within_goal_space = False

            else:
                # print we failed in red colour
                if self.log_internal_state:
                    print("\033[91m" + "Set action failed for --->: " + str(action) + "\033[0m")
                    rospy.logdebug(f"Set action failed for --->: {action}")
                self.movement_result = False
                self.within_goal_space = False

        # --- Joint action
        else:

            # --- Make actions as deltas
            if self.delta_action:

                # we can use smoothing using the action_cycle_time or delta_coeff
                if self.use_smoothing:
                    if self.action_cycle_time == 0.0:
                        # first derivative of the action
                        self.action_vector = self.action_vector + (self.delta_coeff * action)

                        # clip the action vector to be within -1 and 1
                        self.action_vector = np.clip(self.action_vector, -1, 1)

                        # add the action vector to the current ee pos
                        action = self.joint_values + (self.action_vector * self.delta_coeff)

                    else:
                        # first derivative of the action
                        self.action_vector = self.action_vector + (self.action_cycle_time * action)

                        # clip the action vector to be within -1 and 1
                        self.action_vector = np.clip(self.action_vector, -1, 1)

                        # add the action vector to the current ee pos
                        action = self.joint_values + (self.action_vector * self.action_cycle_time)

                else:
                    action = self.joint_values + (action * self.delta_coeff)

            # check if the action is within the joint limits (for the reward calculation)
            min_joint_values = np.array(self.min_joint_values)
            max_joint_values = np.array(self.max_joint_values)
            self.action_not_in_limits = np.any(action <= (min_joint_values + 0.0001)) or np.any(
                action >= (max_joint_values - 0.0001))

            # clip the action
            if self.debug:
                rospy.logwarn(f"Action + current joint_values before clip --->: {action}")

            action = np.clip(action, self.min_joint_values, self.max_joint_values)

            if self.debug:
                rospy.logwarn(f"Action + current joint_values after clip --->: {action}")

            # check if the action is within the workspace
            if self.check_action_within_workspace(action):
                # execute the trajectory - ros_controllers
                self.movement_result = self.move_arm_joints(q_positions=action, time_from_start=self.action_speed)
                self.within_goal_space = True

            else:
                # print we failed in red colour
                if self.log_internal_state:
                    print("\033[91m" + "Set action failed for --->: " + str(action) + "\033[0m")
                    rospy.logdebug(f"Set action failed for --->: {action}")

                self.movement_result = False
                self.within_goal_space = False

    def sample_observation(self):
        """
        Function to get an observation from the environment.

        # traditional observations
        01. EE pos - 3
        02. Vector to the goal (normalized linear distance) - 3
        03. Euclidian distance (ee to reach goal)- 1
        04. Current Joint values - 6
        05. Previous action - 6 or 3 (joint or ee)
        06. Joint velocities - 6

        total: (3x2) + 1 + (6 or 3) + (6x2)

        # depth image
        480x640 32FC1

        # rgb image
        480x640X3 RGB images

        Returns:
            An observation representing the current state of the environment.
        """
        current_goal = self.reach_goal

        # --- 1. Get EE position
        ee_pos_tmp = self.get_ee_pose()  # Get a geometry_msgs/PoseStamped msg
        self.ee_pos = np.array([ee_pos_tmp.pose.position.x, ee_pos_tmp.pose.position.y, ee_pos_tmp.pose.position.z])
        self.ee_ori = np.array([ee_pos_tmp.pose.orientation.x, ee_pos_tmp.pose.orientation.y,
                                ee_pos_tmp.pose.orientation.z, ee_pos_tmp.pose.orientation.w])

        # --- Linear distance to the goal
        linear_dist_ee_goal = current_goal - self.ee_pos  # goal is box dtype and ee_pos is numpy.array. It is okay

        # --- 2. Vector to goal (we are giving only the direction vector)
        vec_ee_goal = linear_dist_ee_goal / np.linalg.norm(linear_dist_ee_goal)

        # --- 3. Euclidian distance
        euclidean_distance_ee_goal = scipy.spatial.distance.euclidean(self.ee_pos, current_goal)  # float

        # --- Get Current Joint values - only for the joints we are using
        #  we need this for delta actions
        # self.joint_values = self.current_joint_positions.copy()  # Get a float list
        self.joint_values = self.get_joint_angles()  # Get a float list
        # we don't need to convert this to numpy array since we concat using numpy below

        if self.prev_action is None:
            # we can use the ee_pos as the previous action - for EE action type
            if self.ee_action_type:
                prev_action = self.ee_pos

            # we can use the joint values as the previous action - for Joint action type
            else:
                prev_action = self.joint_values.copy()
        else:
            prev_action = self.prev_action.copy()

        if self.joint_pos_all is None or self.current_joint_velocities is None:
            done = False
            while not done:
                done = self._check_joint_states_ready()

        # our observations
        obs = np.concatenate((self.ee_pos, vec_ee_goal, euclidean_distance_ee_goal, self.joint_pos_all,
                              prev_action, self.current_joint_velocities), axis=None)

        if self.log_internal_state:
            rospy.loginfo(f"Observations --->: {obs}")

        if self.normal_obs:
            return obs.copy()

        elif self.rgb_obs:
            return self.cv_image_rgb.copy()

        elif self.rgb_plus_normal_obs:
            return {"rgb_image": self.cv_image_rgb.copy(),
                    "observations": obs.copy()}

        elif self.rgb_plus_depth_plus_normal_obs:
            return {"depth_image": self.cv_image_depth.copy(),
                    "rgb_image": self.cv_image_rgb.copy(),
                    "observations": obs.copy()}

        else:
            return obs.copy()

    def calculate_reward(self, achieved_goal=None, desired_goal=None, info=None) -> float:
        """
        Compute the reward for achieving a given goal.

        Sparse Reward: float => 1.0 for success, -1.0 for failure

        Dense Reward:
            if reached: self.reached_goal_reward (positive reward)
            else: - self.mult_dist_reward * distance_to_the_goal

            And as always negative rewards for each step, non-execution and actions not within joint limits

        Args:
            achieved_goal: EE position (optional)
            desired_goal: Reach Goal (optional)
            info (dict): Additional information about the environment. (Optional)

        Returns:
            reward (float): The reward for achieving the given goal.
        """
        # - Init reward
        reward = 0.0

        # check if we are using this with HER
        if info is not None:
            is_her = info.get('is_her', False)
        else:
            is_her = False

        # we don't need to do this but better to check
        if desired_goal is None:
            desired_goal = self.reach_goal

        # check if the achieved_goal is None
        if achieved_goal is None:
            # check if the ee position is None
            if self.ee_pos is not None:
                achieved_goal = self.ee_pos
            else:
                ee_pos_tmp = self.get_ee_pose()  # Get a geometry_msgs/PoseStamped msg
                achieved_goal = np.array(
                    [ee_pos_tmp.pose.position.x, ee_pos_tmp.pose.position.y, ee_pos_tmp.pose.position.z])

        # if it's "Sparse" reward structure
        if self.reward_arc == "Sparse":

            # initialise the sparse reward as negative
            reward = -1.0

            #  if her we don't need to publish the goal marker
            if is_her:
                # check if robot reached the goal
                reach_done = self.check_if_reach_done(achieved_goal, desired_goal)

                if reach_done:
                    reward = 1.0

            else:
                # The marker only turns green if reach is done. Otherwise, it is red.
                self.goal_marker.set_color(r=1.0, g=0.0)
                self.goal_marker.set_duration(duration=5)

                # check if robot reached the goal
                reach_done = self.check_if_reach_done(achieved_goal, desired_goal)

                if reach_done:
                    reward = 1.0

                    # done (green) goal_marker
                    self.goal_marker.set_color(r=0.0, g=1.0)

                # publish the marker to the topic
                self.goal_marker.publish()

                # log the reward
                if self.log_internal_state:
                    rospy.logwarn(">>>REWARD>>>" + str(reward))

        # Since we only look for Sparse or Dense, we don't need to check if it's Dense
        else:

            # # if her we don't need to publish the goal marker, and we always do simple dense reward
            if is_her:
                # - Distance from EE to goal reward
                dist2goal = scipy.spatial.distance.euclidean(achieved_goal, desired_goal)
                reward += - dist2goal

            else:
                # in case of simple dense reward or HER
                if self.simple_dense_reward:
                    # - Distance from EE to goal reward
                    dist2goal = scipy.spatial.distance.euclidean(achieved_goal, desired_goal)
                    reward += - dist2goal

                # for normal dense reward
                else:

                    # - Check if the EE reached the goal
                    done = self.check_if_reach_done(achieved_goal, desired_goal)

                    if done:
                        # EE reached the goal
                        reward += self.reached_goal_reward

                        # done (green) goal_marker
                        self.goal_marker.set_color(r=0.0, g=1.0)
                        self.goal_marker.set_duration(duration=5)

                    else:
                        # not done (red) goal_marker
                        self.goal_marker.set_color(r=1.0, g=0.0)
                        self.goal_marker.set_duration(duration=5)

                        # - Distance from EE to goal reward
                        dist2goal = scipy.spatial.distance.euclidean(achieved_goal, desired_goal)
                        reward += - self.mult_dist_reward * dist2goal

                        # - Constant step reward
                        reward += self.step_reward

                    # publish the goal marker
                    self.goal_marker.publish()

                    # - Check if actions are in limits
                    reward += self.action_not_in_limits * self.joint_limits_reward

                    # - Check if the action is within the goal space
                    reward += (not self.within_goal_space) * self.not_within_goal_space_reward

                    # to punish for actions where we cannot execute
                    if not self.movement_result:
                        reward += self.none_exe_reward

                # log the reward
                if self.log_internal_state:
                    rospy.logwarn(">>>REWARD>>>" + str(reward))

        return reward

    def check_if_done(self):
        """
        Function to check if the episode is done.

        The Task is done if the EE is close enough to the goal

        Returns:
            A boolean value indicating whether the episode has ended
            (e.g. because a goal has been reached or a failure condition has been triggered)
        """
        # this is for logging in different colours
        # Define ANSI escape codes for different colors
        RED = '\033[91m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        BLUE = '\033[94m'
        ENDC = '\033[0m'

        # --- Init done
        done = False

        # - Check if the ee reached the goal
        done_reach = self.check_if_reach_done(self.ee_pos, self.reach_goal)

        if done_reach:
            if self.log_internal_state:
                rospy.loginfo(GREEN + ">>>>>>>>>>>> Reached the Goal! >>>>>>>>>>>" + ENDC)
            done = True

            self.current_action = None  # we don't need to execute any more actions
            self.init_done = False  # we don't need to execute the loop until we reset the env
            self.info_r['is_success'] = True
        else:
            self.info_r['is_success'] = False

        return done

    def check_if_reach_done(self, achieved_goal, desired_goal):
        """
        Check if the reach is done
        """
        done = False

        # distance between achieved goal and desired goal
        distance = scipy.spatial.distance.euclidean(achieved_goal, desired_goal)

        if self.debug:
            print("distance to the goal:", distance)

        if distance <= self.reach_tolerance:
            done = True

        return done

    # not used
    def test_goal_pos(self, goal):
        """
        Function to check if the given goal is reachable
        """
        if self.log_internal_state:
            rospy.logdebug(f"Goal to check: {str(goal)}")
        result = self.check_goal(goal)

        if not result:
            if self.log_internal_state:
                rospy.logdebug("The goal is not reachable")

        return result

    # not used
    def get_random_goal(self, max_tries: int = 100):
        """
        Function to get a reachable goal
        """
        for i in range(max_tries):
            goal = self.goal_space.sample()

            if self.test_goal_pos(goal):
                return True, goal

        if self.log_internal_state:
            rospy.logdebug("Getting a random goal failed!")

        return False, None

    def get_random_goal_no_check(self):
        """
        Function to get a random goal without checking
        """
        return True, self.goal_space.sample()

    # not used
    def check_action_within_goal_space_fk(self, action):
        """
        Function to check if the given action is within the goal space
        """

        # check if the resulting ee pose is within the goal space - using FK
        ee_pos = self.calculate_fk(joint_positions=action)

        if ee_pos is not None:
            # check if the ee pose is within the goal space - using self.goal_space
            if self.goal_space.contains(ee_pos):
                if self.log_internal_state:
                    rospy.logdebug(f"The ee pose of the {action} is within the goal space!")
                return True
            else:
                if self.log_internal_state:
                    rospy.logdebug(f"The ee pose of the {ee_pos} is not within the goal space!")
                return False

        if self.log_internal_state:
            rospy.logwarn("Checking if the action is within the goal space failed!")
        return False

    def check_action_within_workspace(self, action):
        """
        Function to check if the given action is within the workspace

        Args:
            action: The action to be applied to the robot.

        Returns:
            A boolean value indicating whether the action is within the workspace
        """

        BLUE = '\033[94m'
        ENDC = '\033[0m'

        # check if the resulting ee pose is within the workspace - using FK
        ee_pos = self.fk_pykdl(action=action)

        if self.log_internal_state:
            print("goal space", self.workspace_space)  # for debugging

        if ee_pos is not None:
            # check if the ee pose is within the workspace - using self.workspace_space
            if self.workspace_space.contains(ee_pos):
                if self.log_internal_state:
                    rospy.logdebug(f"The ee pose {ee_pos} of the {action} is within the workspace!")
                if self.debug:
                    print(BLUE + f"The ee pose {ee_pos} of the {action} is within the workspace!" + ENDC)
                return True
            else:
                if self.log_internal_state:
                    rospy.logdebug(f"The ee pose {ee_pos} is not within the workspace!")
                if self.debug:
                    print(BLUE + f"The ee pose {ee_pos} is not within the workspace!" + ENDC)
                return False

        if self.log_internal_state:
            rospy.logwarn("Checking if the action is within the workspace failed!")
        if self.debug:
            print(BLUE + "Checking if the action is within the workspace failed!" + ENDC)
        return False

    def check_if_z_within_limits(self, action):
        """
        Function to check if the given ee_pos is within the limits
        """
        # get the ee pose from the action using FK
        ee_pos = self.fk_pykdl(action=action)

        # The robot is mounted on a table. So, we need to check if the z is within the limits
        if ee_pos is not None:

            # check if the z is within the limits
            if ee_pos[2] > self.lowest_z:
                if self.log_internal_state:
                    rospy.logdebug(f"The ee pose {ee_pos} of the {action} is within the z limit!")
                return True
            else:
                if self.log_internal_state:
                    rospy.logdebug(f"The ee pose {ee_pos} is not within the z limit!")
                return False
        else:
            if self.log_internal_state:
                rospy.logwarn("Checking if the action is within the z limit failed!")
            return False

    def _get_params(self):
        """
        Function to get configuration parameters (optional)
        """

        # Action Space
        self.min_joint_values = rospy.get_param('/ned2/min_joint_pos')
        self.max_joint_values = rospy.get_param('/ned2/max_joint_pos')

        # Observation Space
        self.position_ee_max = rospy.get_param('/ned2/position_ee_max')
        self.position_ee_min = rospy.get_param('/ned2/position_ee_min')
        self.rpy_ee_max = rospy.get_param('/ned2/rpy_ee_max')
        self.rpy_ee_min = rospy.get_param('/ned2/rpy_ee_min')
        self.linear_distance_max = rospy.get_param('/ned2/linear_distance_max')
        self.linear_distance_min = rospy.get_param('/ned2/linear_distance_min')
        self.max_distance = rospy.get_param('/ned2/max_distance')
        self.min_joint_vel = rospy.get_param('/ned2/min_joint_vel')
        self.max_joint_vel = rospy.get_param('/ned2/max_joint_vel')
        self.min_joint_angles = rospy.get_param('/ned2/min_joint_angles')
        self.max_joint_angles = rospy.get_param('/ned2/max_joint_angles')

        # Goal space
        self.position_goal_max = rospy.get_param('/ned2/position_goal_max')
        self.position_goal_min = rospy.get_param('/ned2/position_goal_min')

        # Achieved goal
        self.position_achieved_goal_max = rospy.get_param('/ned2/position_achieved_goal_max')
        self.position_achieved_goal_min = rospy.get_param('/ned2/position_achieved_goal_min')

        # Desired goal
        self.position_desired_goal_max = rospy.get_param('/ned2/position_desired_goal_max')
        self.position_desired_goal_min = rospy.get_param('/ned2/position_desired_goal_min')

        # Tolerances
        self.reach_tolerance = rospy.get_param('/ned2/reach_tolerance')

        # Variables related to rewards
        self.step_reward = rospy.get_param('/ned2/step_reward')
        self.mult_dist_reward = rospy.get_param('/ned2/multiplier_dist_reward')
        self.reached_goal_reward = rospy.get_param('/ned2/reached_goal_reward')
        self.joint_limits_reward = rospy.get_param('/ned2/joint_limits_reward')
        self.none_exe_reward = rospy.get_param('/ned2/none_exe_reward')
        self.not_within_goal_space_reward = rospy.get_param('/ned2/not_within_goal_space_reward')

        # workspace
        self.workspace_max = rospy.get_param('/ned2/workspace_max')
        self.workspace_min = rospy.get_param('/ned2/workspace_min')

    # ------------------------------------------------------
    #   Task Methods for launching roscore

    @staticmethod
    def _launch_roscore(port=None, set_new_master_vars=False, default_port=False):
        """
        Launches a new roscore with the specified port. Only updates the ros_port.

        Return:
            ros_port: port of launched roscore
        """

        if port is None:
            port = 11311  # default port

        ros_port = ros_common.launch_roscore(port=port, set_new_master_vars=set_new_master_vars,
                                             default_port=default_port)

        # change to new rosmaster
        ros_common.change_ros_master(ros_port)

        return ros_port
