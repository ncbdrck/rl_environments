#!/bin/python3

from typing import Any, Optional, Dict

import rospy
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register
import scipy.spatial

# Custom robot env
from rl_environments.rx200.sim.robot_envs import rx200_robot_goal_sim_zed2

# core modules of the framework
from multiros.utils import gazebo_core
# from multiros.utils import gazebo_models
# from multiros.utils import gazebo_physics
# from multiros.utils.moveit_multiros import MoveitMultiros
from multiros.utils import ros_common
# from multiros.utils import ros_controllers
from multiros.utils import ros_markers

# Register your environment using the OpenAI register method to utilize gym.make("MyTaskGoalEnv-v0").
register(
    id='RX200Zed2ReacherGoalSim-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_zed2_reach_goal_sim:RX200ReacherGoalEnv',
    max_episode_steps=1000,
)

"""
This is the v0 of the RX200 Reacher Goal conditioned Task Environment. ZED2 camera is used for vision sensing.
- option to use vision sensors - depth and rgb images
- action space is joint positions of the robot arm or xyz position of the end effector. No gripper control
- reward is sparse or dense
- goal is to reach a goal position
- Only works in real-time mode no sequential mode
"""


class RX200ReacherGoalEnv(rx200_robot_goal_sim_zed2.RX200RobotGoalEnv):
    """
    This Task env is for a simple Reach Task with the RX200 robot.
    This env is made to work in a real-time environment.

    The task is done if
        * The robot reached the goal

    Here
        * Action Space - Continuous (5 actions for joints or 3 xyz position of the end effector)
        * Observation - Continuous (28 obs or rgb/depth image or a combination)
        * Desired Goal - Goal we are trying to reach
        * Achieved Goal - Position of the EE

    Init Args:
        * launch_gazebo: Whether to launch Gazebo or not. If False, it is assumed that Gazebo is already running.
        * new_roscore: Whether to launch a new roscore or not. If False, it is assumed that a roscore is already running.
        * roscore_port: Port of the roscore to be launched. If None, a random port is chosen.
        * gazebo_paused: Whether to launch Gazebo in a paused state or not.
        * gazebo_gui: Whether to launch Gazebo with the GUI or not.
        * seed: Seed for the random number generator.
        * reward_type: Type of reward to be used. It Can be "Sparse" or "Dense".
        * delta_action: Whether to use the delta actions or the absolute actions.
        * delta_coeff: Coefficient to be used for the delta actions.
        * ee_action_type: Whether to use the end effector action space or the joint action space.
        * environment_loop_rate: Rate at which the environment should run. (in Hz) default is 10 Hz - (default operating frequency of the robot)
        * action_cycle_time: Time to wait between two consecutive actions. (in seconds) - default 100 ms (should be equal to larger than the environment loop time "1/environment_loop_rate")
        * use_smoothing: Whether to use smoothing for actions or not.
        * rgb_obs_only: Whether to use only the RGB image as the observations or not.
        * normal_obs_only: Whether to use only the traditional observations or not.
        * rgb_plus_normal_obs: Whether to use RGB image and traditional observations or not.
        * rgb_plus_depth_plus_normal_obs: Whether to use RGB image, Depth image and traditional observations or not.
        * load_table: Whether to load the table model or not.
        * debug: Whether to print debug messages or not.
        * log_internal_state: Whether to log the internal state of the environment or not. (action cycle time, etc.)
        * action_speed: set the speed to complete the trajectory. default in 0.5 seconds
        * simple_dense_reward: Whether to use a simple dense reward or not.
    """

    def __init__(self, launch_gazebo: bool = True, new_roscore: bool = True, roscore_port: str = None,
                 gazebo_paused: bool = False, gazebo_gui: bool = False, seed: int = None, reward_type: str = "sparse",
                 delta_action: bool = True, delta_coeff: float = 0.05, ee_action_type: bool = False,
                 environment_loop_rate: float = 10, action_cycle_time: float = 0.800,
                 use_smoothing: bool = False, rgb_obs_only: bool = False, normal_obs_only: bool = True,
                 rgb_plus_normal_obs: bool = False, rgb_plus_depth_plus_normal_obs: bool = False,
                 load_table: bool = True, debug: bool = False, log_internal_state: bool = False,
                 action_speed: float = 0.5, simple_dense_reward: bool = True):

        """
        variables to keep track of ros, gazebo ports and gazebo pid
        """
        ros_port = None
        gazebo_port = None
        gazebo_pid = None

        """
        Initialise the env

        It is recommended to launch Gazebo with a new roscore at this point for the following reasons:,
            1.  This allows running a new rosmaster to enable vectorisation of the environment and the execution of
                multiple environments concurrently.
            2.  The environment can keep track of the process ID of Gazebo to automatically close it when env.close()
                is called.

        """
        # launch gazebo
        if launch_gazebo:

            # Update the function to include additional options.
            ros_port, gazebo_port, gazebo_pid = self._launch_gazebo(launch_roscore=new_roscore, port=roscore_port,
                                                                    paused=gazebo_paused, gui=gazebo_gui)

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
                print("roscore is not running! Launching a new roscore and Gazebo!")
                ros_port, gazebo_port, gazebo_pid = gazebo_core.launch_gazebo(launch_roscore=new_roscore,
                                                                              port=roscore_port,
                                                                              paused=gazebo_paused,
                                                                              gui=gazebo_gui)

        # init the ros node
        if ros_port is not None:
            self.node_name = "RX200ReacherGoalEnvSim" + "_" + ros_port
        else:
            self.node_name = "RX200ReacherGoalEnvSim"

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
        ros_common.ros_load_yaml(pkg_name="rl_environments", file_name="rx200_reach_task_config.yaml", ns="/")
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
        04. Current Joint values - 8
        05. Previous action - 5
        06. Joint velocities - 8
        
        total: (3x2) + 1 + (5 or 3) + (8x2) = 28 or 26
        
        # depth image
        480x640 32FC1
        
        # rgb image
        480x640X3 RGB images

        So observation space is a dictionary with
            observation: Box(28 or 26) or Dict(28 or 26 and 480x640x3 or 480x640)
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

        # Define the depth image space (720x1280 32FC1) - this uses 32-bit float
        self.depth_image_space = spaces.Box(low=0, high=1, shape=(720, 1280), dtype=np.float32)

        # Define the image space (720x1280X3 RGB images) - this uses 8-bit unsigned int
        self.rgb_image_space = spaces.Box(low=0, high=255, shape=(720, 1280, 3), dtype=np.uint8)

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
            use_zed2 = False  # to pass to the superclass
            self.observation_space = spaces.Dict({
                'observation': self.observations,
                'achieved_goal': self.achieved_goal_space,
                'desired_goal': self.desired_goal_space
            })

        elif self.rgb_obs:
            use_zed2 = True
            self.observation_space = spaces.Dict({
                'observation': self.rgb_image_space,
                'achieved_goal': self.achieved_goal_space,
                'desired_goal': self.desired_goal_space
            })

        elif self.rgb_plus_normal_obs:
            use_zed2 = True

            # Define a combined observation space
            obs =  spaces.Dict({
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
            use_zed2 = True
            # Define a combined observation space
            obs =  spaces.Dict({
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
        super().__init__(ros_port=ros_port, gazebo_port=gazebo_port, gazebo_pid=gazebo_pid, seed=seed,
                         real_time=True, action_cycle_time=action_cycle_time, use_zed2=use_zed2,
                         load_table=load_table)

        # for smoothing
        if self.use_smoothing:
            if self.ee_action_type:
                self.action_vector = np.zeros(3, dtype=np.float32)
            else:
                self.action_vector = np.zeros(5, dtype=np.float32)

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
        rospy.loginfo("Initialising the init params!")

        # Initial robot pose - Home
        self.init_pos = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # make the current action None to stop execution for real time envs and also stop the env loop
        self.init_done = False  # we don't need to execute the loop until we reset the env
        self.current_action = None

        # for smoothing
        if self.use_smoothing:
            if self.ee_action_type:
                self.action_vector = np.zeros(3, dtype=np.float32)
            else:
                self.action_vector = np.zeros(5, dtype=np.float32)

        # move the robot to the home pose
        # we need to wait for the movement to finish
        # we define the movement result here so that we can use it in the environment loop (we need it for dense reward)
        self.move_RX200_object.stop_arm()
        self.movement_result = self.move_RX200_object.set_trajectory_joints(self.init_pos)
        if not self.movement_result:
            rospy.logwarn("Homing failed!")

        #  Get a random Reach goal - np.array
        # goal_found, goal_vector = self.get_random_goal()  # this checks if the goal is reachable using moveit
        goal_found, goal_vector = self.get_random_goal_no_check()

        if goal_found:
            self.reach_goal = goal_vector
            rospy.loginfo("Reach Goal--->" + str(self.reach_goal))

        else:
            # fake Reach goal - hard code one
            self.reach_goal = np.array([0.250, 0.000, 0.015], dtype=np.float32)
            rospy.logwarn("Hard Coded Reach Goal--->" + str(self.reach_goal))

        # Publish the goal pos
        self.goal_marker.set_position(position=self.reach_goal)
        self.goal_marker.publish()

        # get initial ee pos and joint values (we need this for delta actions)
        # we don't need this because we reset env just before we start the episode (but just incase)
        ee_pos_tmp = self.get_ee_pose()  # Get a geometry_msgs/PoseStamped msg
        self.ee_pos = np.array([ee_pos_tmp.pose.position.x, ee_pos_tmp.pose.position.y, ee_pos_tmp.pose.position.z])
        self.joint_values = self.get_joint_angles()

        # for dense reward calculation
        self.action_not_in_limits = False
        self.within_goal_space = True

        self.prev_action = self.init_pos.copy()  # for observation

        # We can start the environment loop now

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

        rospy.loginfo("Done resetting the env loop!")

        self.init_done = True
        # self.current_action = self.init_pos.copy()

        rospy.loginfo("Initialising init params done--->")

    def _set_action(self, action):
        """
        Function to apply an action to the robot.

        Args:
            action: Joint positions (numpy array)
        """
        # save the action for observation
        self.prev_action = action.copy()

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
            info (dict): Additional information about the environment.

        Returns:
            reward (float): The reward for achieving the given goal.
        """
        # check if we are using this with HER
        if info is not None:
            is_her = info.get('is_her', False)  # this is set in the HER sampler (my custom implementation)
        else:
            is_her = False

        # Handle multiple goals for HER
        if is_her:
            rospy.loginfo("Using HER reward calculation!")
            rewards = []
            for ag, dg in zip(achieved_goal, desired_goal):
                rewards.append(self.calculate_reward(ag, dg))
            return np.array(rewards)

        reward = None
        if self.reward_r is not None and not is_her:
            reward = self.reward_r

        # incase we don't have a reward yet for real time envs
        # also for Hindsight Experience Replay
        if reward is None:
            reward = self.calculate_reward(achieved_goal, desired_goal, info)

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

        # incase we don't have a done yet for real time envs
        # unnecessary to check since we never set the terminated to None
        if terminated is None:
            terminated = self.check_if_done()

            if terminated:
                self.info["is_success"] = 1.0  # explicitly set the success rate

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
                    rospy.loginfo(f"Executing action --->: {self.action_counter}")
                    self.action_counter += 1
            else:
                self.move_RX200_object.stop_arm()  # stop the arm if there is no action

    def execute_action(self, action):
        """
        Function to apply an action to the robot.

        This method should be implemented here to apply the given action to the robot. The action could be a
        joint position command, a velocity command, or any other type of command that can be applied to the robot.

        Args:
            action: The action to be applied to the robot.
        """
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
                    rospy.logwarn(f"The action: {action} is not reachable!")
                    rospy.logdebug(f"Set action failed for --->: {action}")
                    self.movement_result = False
                    self.within_goal_space = False

            else:
                # print we failed in red colour
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
        04. Current Joint values - 8
        05. Previous action - 5 or 3 (joint or ee)
        06. Joint velocities - 8

        total: (3x2) + 1 + (5 or 3) + (8x2) = 28 or 26

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

        # our observations
        obs = np.concatenate((self.ee_pos, vec_ee_goal, euclidean_distance_ee_goal, self.joint_pos_all,
                              self.prev_action, self.current_joint_velocities), axis=None)

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
            rospy.logwarn(">>>REWARD>>>" + str(reward))

        # Since we only look for Sparse or Dense, we don't need to check if it's Dense
        else:
            # in case of simple dense reward or HER
            if self.simple_dense_reward or is_her:
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
            rospy.loginfo(GREEN + ">>>>>>>>>>>> Reached the Goal! >>>>>>>>>>>" + ENDC)
            done = True

            self.current_action = None  # we don't need to execute any more actions
            self.init_done = False  # we don't need to execute the loop until we reset the env
            self.info_r['is_success'] = 1.0

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
        rospy.logdebug(f"Goal to check: {str(goal)}")
        result = self.check_goal(goal)

        if not result:
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
                rospy.logdebug(f"The ee pose of the {action} is within the goal space!")
                return True
            else:
                rospy.logdebug(f"The ee pose of the {ee_pos} is not within the goal space!")
                return False

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

        if self.debug:
            print("goal space", self.workspace_space)  # for debugging

        if ee_pos is not None:
            # check if the ee pose is within the workspace - using self.workspace_space
            if self.workspace_space.contains(ee_pos):
                rospy.logdebug(f"The ee pose {ee_pos} of the {action} is within the workspace!")
                if self.debug:
                    print(BLUE + f"The ee pose {ee_pos} of the {action} is within the workspace!" + ENDC)
                return True
            else:
                rospy.logdebug(f"The ee pose {ee_pos} is not within the workspace!")
                if self.debug:
                    print(BLUE + f"The ee pose {ee_pos} is not within the workspace!" + ENDC)
                return False

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
                rospy.logdebug(f"The ee pose {ee_pos} of the {action} is within the z limit!")
                return True
            else:
                rospy.logdebug(f"The ee pose {ee_pos} is not within the z limit!")
                return False
        else:
            rospy.logwarn("Checking if the action is within the z limit failed!")
            return False

    def _get_params(self):
        """
        Function to get configuration parameters (optional)
        """

        # Action Space
        self.min_joint_values = rospy.get_param('/rx200/min_joint_pos')
        self.max_joint_values = rospy.get_param('/rx200/max_joint_pos')

        # Observation Space
        self.position_ee_max = rospy.get_param('/rx200/position_ee_max')
        self.position_ee_min = rospy.get_param('/rx200/position_ee_min')
        self.rpy_ee_max = rospy.get_param('/rx200/rpy_ee_max')
        self.rpy_ee_min = rospy.get_param('/rx200/rpy_ee_min')
        self.linear_distance_max = rospy.get_param('/rx200/linear_distance_max')
        self.linear_distance_min = rospy.get_param('/rx200/linear_distance_min')
        self.max_distance = rospy.get_param('/rx200/max_distance')
        self.min_joint_vel = rospy.get_param('/rx200/min_joint_vel')
        self.max_joint_vel = rospy.get_param('/rx200/max_joint_vel')
        self.min_joint_angles = rospy.get_param('/rx200/min_joint_angles')
        self.max_joint_angles = rospy.get_param('/rx200/max_joint_angles')

        # Goal space
        self.position_goal_max = rospy.get_param('/rx200/position_goal_max')
        self.position_goal_min = rospy.get_param('/rx200/position_goal_min')

        # Achieved goal
        self.position_achieved_goal_max = rospy.get_param('/rx200/position_achieved_goal_max')
        self.position_achieved_goal_min = rospy.get_param('/rx200/position_achieved_goal_min')

        # Desired goal
        self.position_desired_goal_max = rospy.get_param('/rx200/position_desired_goal_max')
        self.position_desired_goal_min = rospy.get_param('/rx200/position_desired_goal_min')

        # Tolerances
        self.reach_tolerance = rospy.get_param('/rx200/reach_tolerance')

        # Variables related to rewards
        self.step_reward = rospy.get_param('/rx200/step_reward')
        self.mult_dist_reward = rospy.get_param('/rx200/multiplier_dist_reward')
        self.reached_goal_reward = rospy.get_param('/rx200/reached_goal_reward')
        self.joint_limits_reward = rospy.get_param('/rx200/joint_limits_reward')
        self.none_exe_reward = rospy.get_param('/rx200/none_exe_reward')
        self.not_within_goal_space_reward = rospy.get_param('/rx200/not_within_goal_space_reward')

        # workspace
        self.workspace_max = rospy.get_param('/rx200/workspace_max')
        self.workspace_min = rospy.get_param('/rx200/workspace_min')

    # ------------------------------------------------------
    #   Task Methods for launching gazebo or roscore
    def _launch_gazebo(self, launch_roscore=True, port=None, paused=False, use_sim_time=True,
                       extra_gazebo_args=None, gui=False, recording=False, debug=False,
                       physics="ode", verbose=False, output='screen', respawn_gazebo=False,
                       pub_clock_frequency=100, server_required=False, gui_required=False,
                       custom_world_path=None, custom_world_pkg=None, custom_world_name=None,
                       launch_new_term=True):
        """
        Launches a new Gazebo simulation with the specified options.

        Returns:
            ros_port: None if only launching gazebo and no roscore
            gazebo_port: None if only launching gazebo and no roscore
            gazebo_pid: process id for launched gazebo

        """
        ros_port, gazebo_port, gazebo_pid = gazebo_core.launch_gazebo(
            launch_roscore=launch_roscore,
            port=port,
            paused=paused,
            use_sim_time=use_sim_time,
            extra_gazebo_args=extra_gazebo_args,
            gui=gui,
            recording=recording,
            debug=debug,
            physics=physics,
            verbose=verbose,
            output=output,
            respawn_gazebo=respawn_gazebo,
            pub_clock_frequency=pub_clock_frequency,
            server_required=server_required,
            gui_required=gui_required,
            custom_world_path=custom_world_path,
            custom_world_pkg=custom_world_pkg,
            custom_world_name=custom_world_name,
            launch_new_term=launch_new_term
        )

        return ros_port, gazebo_port, gazebo_pid

    def _launch_roscore(self, port=None, set_new_master_vars=False):
        """
        Launches a new roscore with the specified port. Only updates the ros_port.

        Return:
            ros_port: port of launched roscore
        """

        ros_port, _ = ros_common.launch_roscore(port=int(port), set_new_master_vars=set_new_master_vars)

        # change to new rosmaster
        ros_common.change_ros_master(ros_port)

        return ros_port
