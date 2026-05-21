#!/bin/python3

from typing import Any, Optional, Dict

import rospy
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register
import scipy.spatial
import tf.transformations

# Custom robot env
from rl_environments.ned2.sim.robot_envs import ned2_robot_goal_sim

# core modules of the framework
from multiros.utils import gazebo_core
from multiros.utils import gazebo_models
from multiros.utils import gazebo_physics
from multiros.utils.moveit_multiros import MoveitMultiros
from multiros.utils import ros_common
from multiros.utils import ros_controllers
from multiros.utils import ros_markers

# Register your environment using the gymnasium register method to utilize gym.make("TaskEnv-v0").
# The canonical registration lives in rl_environments/__init__.py — this
# inline register() block is kept (commented) only as a copy-paste reference.
# register(
#     id='NED2PnPGoalSim-v0',
#     entry_point='rl_environments.ned2.sim.task_envs.pnp.ned2_pnp_goal_sim:NED2PnPGoalEnv',
#     max_episode_steps=1000,
# )

"""
NED2 Pick-and-Place Goal Task Environment (sim).

Ported from the RX200 PnP goal sim env (rx200_pnp_goal_sim.py), with the
NED2-specific conventions taken from ned2_pnp_sim.py (the std variant, for
PnP-specific gripper/grasp decisions) and ned2_push_goal_sim.py (for the
goal-env scaffolding: Pose unpacking, GoalEnv hooks, HER safety):
  * 6 arm joints (vs 5 on the RX200) → action dim 6+1=7 (joint mode) or
    3+1=4 (EE mode). The trailing "+1" is the gripper scalar.
  * NED2RobotGoalEnv parent class, ``use_camera`` kwarg (vs ``use_kinect``)
  * SAFETY_CHECK_LINKS live on the robot env and use the ``ned2/...`` URDF
    link names
  * Cube spawn signature is (model_pos_x, model_pos_y); the z is decided
    inside spawn_cube_in_gazebo from ``load_table``
  * Cube model name defaults to "red_cube"
  * Rosparams under '/ned2/...'
  * Gripper API: the NED2 robot env's ``move_gripper_joints`` takes a
    string ("open" / "close") and dispatches via the tools-commander
    action server. The agent still emits a CONTINUOUS scalar gripper
    command in [gripper_min, gripper_max] so the action-space shape stays
    aligned with the RX200 PnP goal env. We discretize at the midpoint
    just before dispatch and ONLY call the action server when the
    discretized state changes vs the previous dispatch (state-change-only
    dispatch — the action server blocks on ``wait_for_result``).
  * is_grasped is derived from the ``joint_base_to_mors_1`` joint state.
    The mors joints are PRISMATIC in metres (URDF: ±0.01 m stroke,
    type="prismatic") — SAME units as RX200's prismatic fingers, just
    symmetric around 0 with a tighter range. The YAML threshold should
    be tuned for NED2's range (≈ 0 m is roughly mid-stroke).
  * ``gripper=True`` is passed to the super-class so the gripper-equipped
    URDF + tools_commander controller are loaded.
  * GoalEnv API: compute_reward / compute_terminated / compute_truncated
    (no underscore) — HER plumbing lives in compute_reward. Cube pose is
    cached in _set_init_params so the first _get_achieved_goal returns a
    real value (mirrors the push goal port).
  * achieved/desired goal are 3-D (cube position) — matching the RX200
    PnP goal template. is_grasped is NOT in the achieved_goal vector; it
    rides in the ``observation`` slot only.
"""


class NED2PnPGoalEnv(ned2_robot_goal_sim.NED2RobotGoalEnv):
    """
    This Task env is for a simple pnp Task with the NED2 robot.

    The task is done if
        * The cube reached the goal

    Here
        * Action Space - Continuous (6 actions for joints or 3 xyz position of the end effector) + 1 gripper scalar
        * Observation - Continuous (obs or rgb/depth image or a combination)

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
        * environment_loop_rate: Rate at which the environment should run. (in Hz) - default 10 Hz (default operating frequency of the robot)
        * action_cycle_time: Time to wait between two consecutive actions. (in seconds) - default 100 ms (should be equal to larger than the environment loop time "1/environment_loop_rate")
        * realtime_mode: If True (default), runs the UniROS paper §7 real-time loop — physics is never paused, a rospy.Timer at ``environment_loop_rate`` updates obs/reward/done, and ``step()`` reads the latest cached values. This matches the real env, so policies transfer / concurrent sim+real learning Just Works. If False, runs the standard MDP loop — Gazebo physics is paused around each ``_set_action``, the action is executed synchronously, the agent waits ``action_cycle_time`` for the trajectory, then a fresh obs/reward/done is sampled. The non-realtime mode is for clean RL-algorithm benchmarking where you want every sample to correspond exactly to the post-action world state.
        * use_smoothing: Whether to use smoothing for actions or not.
        * rgb_obs_only: Whether to use only the RGB image as the observations or not.
        * normal_obs_only: Whether to use only the traditional observations or not.
        * rgb_plus_normal_obs: Whether to use RGB image and traditional observations or not.
        * rgb_plus_depth_plus_normal_obs: Whether to use RGB image, Depth image and traditional observations or not.
        * load_table: Whether to load the table model or not.
        * debug: Whether to print debug messages or not.
        * action_speed: set the speed to complete the trajectory. default in 0.5 seconds
        * simple_dense_reward: Whether to use a simple dense reward or not.
        * log_internal_state: Whether to log the internal state of the environment or not.
        * random_goal: Whether to use a random goal or not.
        * random_cube_spawn: Whether to spawn the cube at a random position or not.
        * multi_goal: Whether to use the multi-goal curriculum (intermediate lift goal then final pnp goal).

    """

    def __init__(self, launch_gazebo: bool = True, new_roscore: bool = True, roscore_port: str = None,
                 gazebo_paused: bool = False, gazebo_gui: bool = False, seed: int = None, reward_type: str = "Dense",
                 delta_action: bool = True, delta_coeff: float = 0.05, ee_action_type: bool = False,
                 environment_loop_rate: float = 25, action_cycle_time: float = 0.100,
                 use_smoothing: bool = False, rgb_obs_only: bool = False, normal_obs_only: bool = True,
                 rgb_plus_normal_obs: bool = False, rgb_plus_depth_plus_normal_obs: bool = False,
                 load_table: bool = True, debug: bool = False, action_speed: float = 0.5,
                 simple_dense_reward: bool = True, log_internal_state: bool = False, random_goal: bool = True,
                 random_cube_spawn: bool = True,
                 realtime_mode: bool = True,
                 multi_goal: bool = False, use_wrist_camera: bool = False):

        # Real-time vs normal MDP step mode. See docstring above.
        self.realtime_mode = realtime_mode

        # Multi-goal curriculum: when True, the env emits an intermediate
        # "lift the cube" goal (cube_spawn_pos + [0, 0, lift_height])
        # until reached, then switches to the final pnp_goal. The agent's
        # desired_goal in obs switches alongside so HER stays consistent.
        # check_if_done always uses the FINAL goal — never terminates on
        # the intermediate.
        self.multi_goal = multi_goal

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
            self.node_name = "NED2PnPGoalEnv" + "_" + ros_port
        else:
            self.node_name = "NED2PnPGoalEnv"

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
        Goal and Cube spawn
        """
        self.random_goal = random_goal
        self.random_cube_spawn = random_cube_spawn

        """
        Debug
        """
        self.debug = debug

        """
        Load YAML param file
        """

        # add to ros parameter server
        ros_common.ros_load_yaml(pkg_name="rl_environments", file_name="ned2_pnp_task_config.yaml", ns="/")
        self._get_params()

        """
        Define the action space.
        """
        # Joint action space or End effector action space
        # ROS and Gazebo often use double-precision (64-bit),
        # but we are using single-precision (32-bit) as it is typical for RL implementations.

        # PnP needs gripper control on top of arm motion. Action space is
        # extended by ONE scalar gripper command. NED2's gripper API is
        # binary (open/close action server), but we keep the scalar slot
        # so the action space matches RX200 PnP goal — the discretization
        # happens inside execute_action right before the action server
        # call.
        if self.ee_action_type:
            # EE-pos (3) + gripper (1) = 4 DOF
            self.max_ee_values = np.array([self.position_ee_max["x"], self.position_ee_max["y"],
                                           self.position_ee_max["z"]])
            self.min_ee_values = np.array([self.position_ee_min["x"], self.position_ee_min["y"],
                                           self.position_ee_min["z"]])

            low = np.concatenate([self.min_ee_values, [self.gripper_min]]).astype(np.float32)
            high = np.concatenate([self.max_ee_values, [self.gripper_max]]).astype(np.float32)
            self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        else:
            # 6 arm joints + gripper = 7 DOF (NED2 has 6 arm joints)
            low = np.concatenate([np.array(self.min_joint_values), [self.gripper_min]]).astype(np.float32)
            high = np.concatenate([np.array(self.max_joint_values), [self.gripper_max]]).astype(np.float32)
            self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Midpoint of the gripper scalar range — used to discretize the
        # continuous command into "open" / "close" before dispatching to
        # the NED2 tools-commander action server.
        self.gripper_cmd_mid = 0.5 * (self.gripper_min + self.gripper_max)

        """
        Define the observation space.

        # typical observation
        01. EE pos - 3
        02. EE rpy - 3
        03. Vector to the goal (normalized linear distance) - 3
        04. Euclidian distance (cube to pnp goal)- 1
        05. Current Joint values - len(joint_pos_all)  (6 arm + gripper joints)
        06. Previous action - 7 or 4 (joint+gripper or ee+gripper)
        07. Joint velocities - same len as joint_pos_all
        08. Cube pos - 3
        09. Cube rpy - 3
        10. Cube linear velocity (finite-diff) - 3
        11. Cube angular velocity (rpy-diff) - 3
        12. Cube position relative to EE - 3
        13. is_grasped - 1 (derived binary)

        # depth image
        480x640 32FC1

        # rgb image
        480x640X3 RGB images
        """

        # ---- ee pos
        observations_high_ee_pos_range = np.array(
            np.array([self.position_ee_max["x"], self.position_ee_max["y"], self.position_ee_max["z"]]))
        observations_low_ee_pos_range = np.array(
            np.array([self.position_ee_min["x"], self.position_ee_min["y"], self.position_ee_min["z"]]))

        # ----- ee rpy
        observations_high_ee_rpy = np.array(
            np.array([self.rpy_ee_max["r"], self.rpy_ee_max["p"], self.rpy_ee_max["y"]]))
        observations_low_ee_rpy = np.array(
            np.array([self.rpy_ee_min["r"], self.rpy_ee_min["p"], self.rpy_ee_min["y"]]))

        # ---- vector to the goal - normalized linear distance
        observations_high_vec_ee_goal = np.array([1.0, 1.0, 1.0])
        observations_low_vec_ee_goal = np.array([-1.0, -1.0, -1.0])

        # ---- Euclidian distance
        observations_high_dist = np.array([self.max_distance])
        observations_low_dist = np.array([0.0])

        # ---- joint values
        observations_high_joint_values = self.max_joint_angles.copy()
        observations_low_joint_values = self.min_joint_angles.copy()

        # ---- previous action (extended by 1 dim for the gripper scalar so
        # the obs prev_action matches the 7-DOF / 4-DOF action_space).
        if self.ee_action_type:
            observations_high_prev_action = np.concatenate(
                [self.max_ee_values, [self.gripper_max]]).astype(np.float32)
            observations_low_prev_action = np.concatenate(
                [self.min_ee_values, [self.gripper_min]]).astype(np.float32)
        else:
            observations_high_prev_action = np.concatenate(
                [np.array(self.max_joint_values), [self.gripper_max]]).astype(np.float32)
            observations_low_prev_action = np.concatenate(
                [np.array(self.min_joint_values), [self.gripper_min]]).astype(np.float32)

        # ---- joint velocities
        observations_high_joint_vel = self.max_joint_vel.copy()
        observations_low_joint_vel = self.min_joint_vel.copy()

        # ---- cube pos
        observations_high_cube_pos = np.array(
            np.array([self.position_cube_max["x"], self.position_cube_max["y"], self.position_cube_max["z"]]))
        observations_low_cube_pos = np.array(
            np.array([self.position_cube_min["x"], self.position_cube_min["y"], self.position_cube_min["z"]]))

        # ---- cube rpy
        observations_high_cube_rpy = np.array(
            np.array([self.rpy_cube_max["r"], self.rpy_cube_max["p"], self.rpy_cube_max["y"]]))
        observations_low_cube_rpy = np.array(
            np.array([self.rpy_cube_min["r"], self.rpy_cube_min["p"], self.rpy_cube_min["y"]]))

        # ---- cube linear velocity (finite-diff)
        observations_high_cube_lin_vel = np.array(
            np.array([self.linear_velocity_cube_max["x"], self.linear_velocity_cube_max["y"],
                      self.linear_velocity_cube_max["z"]]))
        observations_low_cube_lin_vel = np.array(
            np.array([self.linear_velocity_cube_min["x"], self.linear_velocity_cube_min["y"],
                      self.linear_velocity_cube_min["z"]]))

        # ---- cube angular velocity (rpy-diff with wrap-around)
        observations_high_cube_ang_vel = np.array(
            np.array([self.angular_velocity_cube_max["r"], self.angular_velocity_cube_max["p"],
                      self.angular_velocity_cube_max["y"]]))
        observations_low_cube_ang_vel = np.array(
            np.array([self.angular_velocity_cube_min["r"], self.angular_velocity_cube_min["p"],
                      self.angular_velocity_cube_min["y"]]))

        # ---- cube position relative to EE (cube_pos - ee_pos)
        observations_high_cube_rel = np.array(
            np.array([self.cube_rel_to_ee_max["x"], self.cube_rel_to_ee_max["y"],
                      self.cube_rel_to_ee_max["z"]]))
        observations_low_cube_rel = np.array(
            np.array([self.cube_rel_to_ee_min["x"], self.cube_rel_to_ee_min["y"],
                      self.cube_rel_to_ee_min["z"]]))

        # ---- is_grasped (derived binary, 1 dim)
        observations_high_is_grasped = np.array([1.0])
        observations_low_is_grasped = np.array([0.0])

        high = np.concatenate(
            [observations_high_ee_pos_range, observations_high_ee_rpy, observations_high_vec_ee_goal,
             observations_high_dist, observations_high_joint_values, observations_high_prev_action,
             observations_high_joint_vel, observations_high_cube_pos, observations_high_cube_rpy,
             observations_high_cube_lin_vel, observations_high_cube_ang_vel, observations_high_cube_rel,
             observations_high_is_grasped, ])

        low = np.concatenate(
            [observations_low_ee_pos_range, observations_low_ee_rpy, observations_low_vec_ee_goal,
             observations_low_dist, observations_low_joint_values, observations_low_prev_action,
             observations_low_joint_vel, observations_low_cube_pos, observations_low_cube_rpy,
             observations_low_cube_lin_vel, observations_low_cube_ang_vel, observations_low_cube_rel,
             observations_low_is_grasped, ])

        # Define the traditional observation space
        self.observations = spaces.Box(low=low, high=high, dtype=np.float32)

        # Define the depth image space (480x640 32FC1) - this uses 32-bit float
        self.depth_image_space = spaces.Box(low=0, high=1, shape=(480, 640), dtype=np.float32)

        # Define the image space (480x640X3 RGB images) - this uses 8-bit unsigned int
        self.rgb_image_space = spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8)

        # Goal-conditioned bounds (achieved = current cube pos, desired = pnp goal).
        # achieved-goal bounds match the cube tracking envelope; desired-goal
        # bounds match the sampling range so HER-relabeled samples stay in-space.
        # NOTE: matches the RX200 PnP goal template's 3-D achieved goal —
        # is_grasped rides in the ``observation`` slot, NOT the achieved_goal,
        # so HER relabeling stays on cube-position-only.
        achieved_goal_high = np.array([self.position_achieved_goal_max["x"],
                                       self.position_achieved_goal_max["y"],
                                       self.position_achieved_goal_max["z"]], dtype=np.float32)
        achieved_goal_low = np.array([self.position_achieved_goal_min["x"],
                                      self.position_achieved_goal_min["y"],
                                      self.position_achieved_goal_min["z"]], dtype=np.float32)
        desired_goal_high = np.array([self.position_desired_goal_max["x"],
                                      self.position_desired_goal_max["y"],
                                      self.position_desired_goal_max["z"]], dtype=np.float32)
        desired_goal_low = np.array([self.position_desired_goal_min["x"],
                                     self.position_desired_goal_min["y"],
                                     self.position_desired_goal_min["z"]], dtype=np.float32)
        self.achieved_goal_space = spaces.Box(low=achieved_goal_low, high=achieved_goal_high, dtype=np.float32)
        self.desired_goal_space = spaces.Box(low=desired_goal_low, high=desired_goal_high, dtype=np.float32)

        # Define the final observation space.
        # GazeboGoalEnv.step assembles an outer Dict with three keys —
        # ``observation``, ``achieved_goal``, ``desired_goal``. The
        # ``observation`` slot is either a flat array (normal_obs / rgb_obs)
        # or a nested sensor Dict when multiple sensors are active. Mirrors
        # the reach_goal scheme so HER + SB3 sees identical shapes across
        # tasks. NED2RobotGoalEnv exposes ``use_camera`` (not
        # ``use_kinect``) for the head-mount-kinect2 stream — keep the
        # local variable name aligned with that kwarg.
        if self.normal_obs:
            use_camera = False
            self.observation_space = spaces.Dict({
                'observation': self.observations,
                'achieved_goal': self.achieved_goal_space,
                'desired_goal': self.desired_goal_space,
            })

        elif self.rgb_obs:
            use_camera = True
            self.observation_space = spaces.Dict({
                'observation': self.rgb_image_space,
                'achieved_goal': self.achieved_goal_space,
                'desired_goal': self.desired_goal_space,
            })

        elif self.rgb_plus_normal_obs:
            use_camera = True
            _inner = spaces.Dict({
                "rgb_image": self.rgb_image_space,
                "observations": self.observations,
            })
            self.observation_space = spaces.Dict({
                'observation': _inner,
                'achieved_goal': self.achieved_goal_space,
                'desired_goal': self.desired_goal_space,
            })

        elif self.rgb_plus_depth_plus_normal_obs:
            use_camera = True
            _inner = spaces.Dict({
                "depth_image": self.depth_image_space,
                "rgb_image": self.rgb_image_space,
                "observations": self.observations,
            })
            self.observation_space = spaces.Dict({
                'observation': _inner,
                'achieved_goal': self.achieved_goal_space,
                'desired_goal': self.desired_goal_space,
            })

        else:
            use_camera = False
            self.observation_space = spaces.Dict({
                'observation': self.observations,
                'achieved_goal': self.achieved_goal_space,
                'desired_goal': self.desired_goal_space,
            })

        """
        Goal space for sampling
        - default - not used for selecting a random goal
        - used for spawning the cube at a random position in Gazebo - random_cube_spawn==True
        - if specified, sample a goal within the specified range to push the cube to - random_goal==True
        """
        # ---- Goal pos
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
        self.cube_marker = ros_markers.RosMarker(frame_id="world", ns="cube", marker_type=2, marker_topic="cube_pos",
                                                 lifetime=30.0)

        """
        Init super class.
        """
        # NED2RobotGoalEnv maps real_time → unpause_pause_physics; this
        # single flag drives both step modes. NED2 uses ``use_camera`` (not
        # ``use_kinect``) for the head-mount-kinect2 stream. PnP needs the
        # gripper, so ``gripper=True`` is passed so the gripper-equipped
        # URDF + tools_commander controller are loaded.
        super().__init__(ros_port=ros_port, gazebo_port=gazebo_port, gazebo_pid=gazebo_pid, seed=seed,
                         real_time=self.realtime_mode, action_cycle_time=action_cycle_time, use_camera=use_camera, use_wrist_camera=use_wrist_camera,
                         load_table=load_table, gripper=True)

        # for smoothing — NED2 arm dim is 6 (vs RX200's 5).
        if self.use_smoothing:
            if self.ee_action_type:
                self.action_vector = np.zeros(3, dtype=np.float32)
            else:
                self.action_vector = np.zeros(6, dtype=np.float32)

        # we can use this to set a time for ros_controllers to complete the action
        self.environment_loop_time = 1.0 / environment_loop_rate  # in seconds

        self.prev_action = None  # for observation

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

            # Real-time mode only: spin up the rospy.Timer-driven env loop
            # (paper §7). Normal mode reuses the same cache but the compute
            # happens synchronously inside _set_action — no timer.
            if self.realtime_mode:
                rospy.Timer(rospy.Duration(1.0 / environment_loop_rate), self.environment_loop)

        # for dense reward calculation
        self.action_not_in_limits = False
        self.lowest_z = self.workspace_min["z"]  # lowest z value in the workspace

        # Cube velocity state (finite-diff baseline). Reset to None at the
        # start of each episode in _set_init_params so the first tick of
        # a new episode reads zero velocity rather than a spurious value
        # carried over from the previous episode's last pose.
        self.prev_cube_pos = None
        self.prev_cube_rpy = None
        self.prev_cube_time = None
        self.cube_linear_velocity = np.zeros(3, dtype=np.float32)
        self.cube_angular_velocity = np.zeros(3, dtype=np.float32)

        # Grasp / multi-goal trackers (PnP-specific).
        # ``is_grasped`` is derived in sample_observation from cube-rel-to-EE
        # distance + ``joint_base_to_mors_1`` position; kept as float so it
        # can ride in the obs vector with the same dtype as everything else.
        self.is_grasped = 0.0
        # ``intermediate_goal`` is set in _set_init_params when multi_goal
        # is True; held as None otherwise so accidental reads surface.
        self.intermediate_goal = None
        self.intermediate_reached = False
        # Track the most recent discretized gripper command so we don't
        # spam the tools-commander action server with redundant goals
        # every env loop tick (the action server is action-based and
        # waits for completion).
        self.last_gripper_state = None  # None | "open" | "close"
        self.movement_result = False
        self.within_goal_space = False

        # Cube pose cache so _get_achieved_goal has something to return
        # before the first sample_observation tick has populated it.
        # Seeded in _set_init_params with the spawn pose; mirrors the
        # push goal port's first-reset HER safety pattern.
        self.cube_pos = None
        self.cube_ori = None

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
            2. Open the gripper (PnP episodes start with an open gripper)
            3. Find a valid random position to spawn cube
            4. Spawn the cube in Gazebo
            5. Find a random goal position to pnp the cube to if random_goal is True
            6. Publish the goal position and cube position as markers

        """
        if self.log_internal_state:
            rospy.loginfo("Initialising the init params!")

        # --------------- Initial robot pose - Home (NED2 has 6 arm joints).
        self.init_pos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Reset cube-velocity finite-diff state so the new episode doesn't
        # inherit the previous one's tail-end pose as its baseline.
        self.prev_cube_pos = None
        self.prev_cube_rpy = None
        self.prev_cube_time = None
        self.cube_linear_velocity = np.zeros(3, dtype=np.float32)
        self.cube_angular_velocity = np.zeros(3, dtype=np.float32)

        # Reset grasp / multi-goal state. intermediate_goal is computed
        # later in this method, after the cube has been (re)spawned so
        # cube_pos reflects the actual starting pose. multi_goal=False
        # leaves intermediate_goal=None and intermediate_reached untouched.
        self.is_grasped = 0.0
        self.intermediate_goal = None
        self.intermediate_reached = False
        self.last_gripper_state = None  # force a fresh open at episode start

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
        # Open the gripper at episode start so the agent always begins with
        # an empty hand. NED2's gripper API is the tools-commander action
        # server — string commands only.
        try:
            self.move_gripper_joints("open")
            self.last_gripper_state = "open"
        except Exception as _e:
            if self.log_internal_state:
                rospy.logwarn(f"[PNP] init gripper open failed: {_e}")
        if not self.movement_result:
            if self.log_internal_state:
                rospy.logwarn("Homing failed!")

        # --------------- Remove and Spawn the cube
        # remove the cube
        self.remove_cube_in_gazebo()

        # spwan the cube
        if self.random_cube_spawn:
            #  Get a random pos - np.array
            cube_init_vector = self.get_random_cube_init_pose()

        # if we don't spwan cube randomly, we can hard code one
        else:
            # Static cube position - hard code one
            # TODO: confirm NED2 pnp static cube spawn pose
            cube_init_vector = np.array([0.25, 0.00, 0.015], dtype=np.float32)

        # spawn the cube. The NED2 robot env signature is
        # spawn_cube_in_gazebo(model_pos_x, model_pos_y) — it picks z
        # internally based on ``load_table``.
        self.spawn_cube_in_gazebo(model_pos_x=cube_init_vector[0],
                                  model_pos_y=cube_init_vector[1])
        if self.log_internal_state:
            rospy.logwarn("Hard Coded Cube init pos--->" + str(cube_init_vector))

        # Publish the cube pos
        self.cube_marker.set_position(position=cube_init_vector)  #  so we can see the cube in rviz
        self.cube_marker.publish()

        # --------------- Random PnP Goal
        if self.random_goal:
            #  Get a random pos - np.array
            self.pnp_goal = self.get_random_goal_no_check()

        # if we don't have a random pnp goal, we can hard code one
        else:
            # fake pnp goal - hard code one
            # We don't need to worry if we are using a table or not since we get cube pos wrt to base_link
            # TODO: confirm NED2 pnp static goal pose
            self.pnp_goal = np.array([0.250, 0.000, 0.015], dtype=np.float32)


        if self.log_internal_state:
            rospy.logwarn("Hard Coded PnP Goal--->" + str(self.pnp_goal))

        # Publish the goal pos
        self.goal_marker.set_position(position=self.pnp_goal)
        self.goal_marker.publish()

        # Seed the cube pose cache with the spawn pose so _get_achieved_goal
        # has something real to return before the first sample_observation
        # tick — otherwise reset()'s obs dict would carry achieved_goal=zeros,
        # which silently corrupts HER's first relabeled sample. Mirrors the
        # push goal port's first-reset HER safety pattern.
        self.cube_pos = np.asarray(cube_init_vector, dtype=np.float32).copy()
        self.cube_ori = np.zeros(3, dtype=np.float32)

        # Multi-goal: compute the intermediate lift target from the
        # current cube pose. cube_pos was just seeded above from the spawn
        # vector, so the intermediate target is anchored to the actual
        # starting pose rather than a stale or zero value.
        if self.multi_goal:
            cube_xyz = (np.asarray(self.cube_pos, dtype=np.float32)
                        if getattr(self, "cube_pos", None) is not None
                        else np.array([0.25, 0.0, 0.015], dtype=np.float32))
            self.intermediate_goal = cube_xyz + np.array(
                [0.0, 0.0, float(self.lift_height)], dtype=np.float32)

        #  --------------- Set init values for reward calculation and observation
        # get initial ee pos and joint values (we need this for delta actions or when we have EE action space)
        ee_pos_tmp = self.get_ee_pose()  # Get a geometry_msgs/PoseStamped msg
        self.ee_pos = np.array([ee_pos_tmp.pose.position.x, ee_pos_tmp.pose.position.y, ee_pos_tmp.pose.position.z])
        self.ee_ori = np.array([ee_pos_tmp.pose.orientation.x, ee_pos_tmp.pose.orientation.y,
                               ee_pos_tmp.pose.orientation.z, ee_pos_tmp.pose.orientation.w])  # for IK calculation - EE actions
        self.joint_values = self.get_joint_angles()

        # for dense reward calculation
        self.action_not_in_limits = False
        self.within_goal_space = True

        # Episode-start prev_action: arm portion (ee_pos or init_pos) +
        # gripper at its open value (gripper_max). PnP episodes start with
        # the gripper open; the agent commands close when contacting the
        # cube.
        if self.ee_action_type:
            self.prev_action = np.concatenate(
                [self.ee_pos, [self.gripper_max]]).astype(np.float32)
        else:
            self.prev_action = np.concatenate(
                [self.init_pos, [self.gripper_max]]).astype(np.float32)

        #  --------------- Set init values for the environment loop
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

        Real-time mode (default): stash the action; the timer-driven
        environment_loop is what calls execute_action (paper §7).
        Normal MDP mode (realtime_mode=False): execute the action
        synchronously and clear the obs/reward/done cache so the
        _get_* fallbacks resample against the post-action world after
        GazeboBaseEnv.step's action_cycle_time sleep.

        Args:
            action: The action to be applied to the robot.
        """
        # save the action for observation
        self.prev_action = action.copy()

        if self.log_internal_state:
            rospy.loginfo(f"Applying action---> {action}")

        self.current_action = action.copy()

        if not self.realtime_mode:
            self.obs_r = None
            self.reward_r = None
            self.terminated_r = None
            self.info_r = {}
            self.execute_action(action)

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

        # incase we don't have an observation yet
        if obs is None:
            obs = self.sample_observation()

        return obs.copy()

    # GoalEnv step() calls compute_reward / compute_terminated / compute_truncated
    # (no underscore) instead of the BaseEnv underscore hooks. Those are defined
    # below alongside the HER plumbing — no _get_reward / _compute_terminated /
    # _compute_truncated needed here.

    # -------------------------------------------------------
    #   Include any custom methods available for the MyTaskEnv class

    def environment_loop(self, event):
        """
        Function for Environment loop for real time RL envs
        """

        #  we don't need to execute the loop until we reset the env
        if self.init_done:

            # Close-race guard (see reach v3 0712f5a for the diagnosis):
            # rospy.Timer keeps firing during env.close() while MoveIt
            # cleanup runs its 1s wait_for_message timeouts. Controllers
            # get unspawned mid-close → joint_states stops → get_joint_angles
            # returns []. Next tick's execute_action crashes on the
            # delta-action broadcast (shape (0,) vs (6,)). Bail out cleanly
            # if ROS is shutting down or joint state is stale. NED2 has 6
            # arm joints.
            if rospy.is_shutdown():
                return
            jv = getattr(self, "joint_values", None)
            if jv is None or len(jv) < 6:
                return

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

        PnP action layout:
          * Joint mode: ``action[:6]`` = 6 arm joint commands (delta or
            absolute per ``delta_action``); ``action[6]`` = gripper scalar.
          * EE mode:    ``action[:3]`` = EE position (delta or absolute);
            ``action[3]`` = gripper scalar.
        The gripper scalar is ALWAYS treated as an absolute command in
        [gripper_min, gripper_max] regardless of ``delta_action`` (open /
        close is closer to a discrete decision than a small delta; matches
        FetchPickAndPlace's discrete-ish gripper convention).

        NED2's gripper is driven by a tools-commander action server that
        only accepts "open" / "close" — we discretize the scalar at the
        midpoint (``gripper_cmd_mid``) just before dispatch. To keep the
        action server from being called on every env-loop tick, we only
        send a new goal when the discretized state CHANGES vs the last
        state we dispatched (``last_gripper_state``).

        Args:
            action: The 7-DOF (joint) or 4-DOF (EE) action vector.
        """
        if self.log_internal_state:
            rospy.loginfo(f"Action --->: {action}")

        # Split off the gripper command up front. The arm-portion code
        # below is identical to push goal; we just operate on the sliced
        # arm action so existing delta / clip / safety logic keeps working
        # unchanged.
        action = np.asarray(action, dtype=np.float32)
        gripper_cmd = float(np.clip(action[-1], self.gripper_min, self.gripper_max))
        action = action[:-1]

        # --- Set the action based on the action type
        # --- EE action
        if self.ee_action_type:

            # --- Get the current EE position
            ee_pos_tmp = self.get_ee_pose()  # Get a geometry_msgs/PoseStamped msg
            self.ee_pos = np.array([ee_pos_tmp.pose.position.x, ee_pos_tmp.pose.position.y, ee_pos_tmp.pose.position.z])

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
                    # Per-link FK safety (NED2RobotEnv._check_action_links_safe).
                    # Workspace + IK-feasible doesn't mean every link stays
                    # above the table — shoulder/elbow/wrist can dip below
                    # while EE target sits above. See reach v3 (0a6dfb3) for
                    # the full rationale.
                    safe, reason = self._check_action_links_safe(
                        joint_positions, current_joints=self.joint_values
                    )
                    if not safe:
                        if self.log_internal_state:
                            rospy.logwarn(f"[SAFETY] EE action rejected: {reason}")
                        self.movement_result = False
                        self.within_goal_space = False
                    else:
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
                if self.log_internal_state:
                    rospy.logdebug(f"Set action failed for --->: {action}")
                self.movement_result = False
                self.within_goal_space = False

        # --- Joint action
        else:

            # --- Make actions as deltas
            if self.delta_action:

                # get the current joint values
                self.joint_values = self.get_joint_angles()

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
                if self.log_internal_state:
                    rospy.logwarn(f"Action + current joint_values before clip --->: {action}")

            action = np.clip(action, self.min_joint_values, self.max_joint_values)

            if self.debug:
                if self.log_internal_state:
                    rospy.logwarn(f"Action + current joint_values after clip --->: {action}")

            # check if the action is within the workspace
            if self.check_action_within_workspace(action):
                # Per-link FK safety. self.joint_values was refreshed at the
                # top of this delta-action block, so the delta cap can use it.
                safe, reason = self._check_action_links_safe(
                    action, current_joints=self.joint_values
                )
                if not safe:
                    if self.log_internal_state:
                        rospy.logwarn(f"[SAFETY] joint action rejected: {reason}")
                    self.movement_result = False
                    self.within_goal_space = False
                else:
                    # execute the trajectory - ros_controllers
                    self.movement_result = self.move_arm_joints(q_positions=action, time_from_start=self.action_speed)
                    self.within_goal_space = True

            else:
                if self.log_internal_state:
                    rospy.logdebug(f"Set action failed for --->: {action}")
                self.movement_result = False
                self.within_goal_space = False

        # Gripper command: discretize the continuous scalar to the NED2
        # tools-commander API. Only dispatch when the discrete state
        # CHANGES vs the previous dispatch — the action server blocks on
        # ``wait_for_result``, so spamming "open" / "close" every tick
        # would dominate the loop time.
        desired_gripper_state = "close" if gripper_cmd < self.gripper_cmd_mid else "open"
        if desired_gripper_state != self.last_gripper_state:
            try:
                self.move_gripper_joints(desired_gripper_state)
                self.last_gripper_state = desired_gripper_state
            except Exception as _e:
                if self.log_internal_state:
                    rospy.logwarn(f"[PNP] gripper command failed: {_e}")

    def sample_observation(self):
        """
        Function to get an observation from the environment.

        # traditional observations
        01. EE pos - 3
        02. EE rpy - 3
        03. Vector to the goal (normalized linear distance) - 3
        04. Euclidian distance (cube to pnp goal)- 1
        05. Current Joint values - len(joint_pos_all)  (6 arm + gripper joints)
        06. Previous action - 7 or 4 (joint+gripper or ee+gripper)
        07. Joint velocities - same len as joint_pos_all
        08. Cube pos - 3
        09. Cube rpy - 3
        10. Cube linear velocity (finite-diff) - 3
        11. Cube angular velocity (rpy-diff) - 3
        12. Cube position relative to EE - 3
        13. is_grasped - 1 (derived binary)

        # depth image
        480x640 32FC1

        # rgb image
        480x640X3 RGB images

        Returns:
            An observation representing the current state of the environment.
        """
        # --- Get the current ACTIVE goal: intermediate (lift) target while
        # we're in the pre-grasp phase under multi_goal, else the final
        # pnp_goal. Keeps vec_ee_goal / euclidean_distance / dense reward
        # all consistent with whatever the agent is currently chasing.
        if (self.multi_goal and not self.intermediate_reached
                and self.intermediate_goal is not None):
            current_goal = self.intermediate_goal
        else:
            current_goal = self.pnp_goal

        # --- Get the current cube position and orientation.
        # GoalEnv variant of the NED2 robot env returns a single
        # geometry_msgs/Pose (not the 3-tuple the standard robot env
        # gives). Adapt it inline here: pose=None means lookup failed.
        # The NED2 robot env's get_model_pose defaults to
        # model_name="red_cube".
        cube_pose_msg = self.get_model_pose()
        if cube_pose_msg is None:
            cube_pose_done = False
        else:
            cube_pose_done = True
            self.cube_pos = np.array([cube_pose_msg.position.x,
                                      cube_pose_msg.position.y,
                                      cube_pose_msg.position.z], dtype=np.float32)
            self.cube_ori = np.array(tf.transformations.euler_from_quaternion([
                cube_pose_msg.orientation.x, cube_pose_msg.orientation.y,
                cube_pose_msg.orientation.z, cube_pose_msg.orientation.w]),
                dtype=np.float32)

        # if the cube pose is not found, we can set the current cube pos to 0
        # we need to set this to 0 so that we can get the observations
        if not cube_pose_done:
            if self.log_internal_state:
                rospy.logwarn("Cube pose not found!")
            self.cube_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self.cube_ori = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # publish the cube pos marker
        self.cube_marker.set_position(position=self.cube_pos)
        self.cube_marker.set_color(r=0.0, g=0.0, b=1.0)  # let's make marker colour blue
        self.cube_marker.set_duration(duration=5)
        self.cube_marker.publish()

        # --- 1. Get EE position
        ee_pos_tmp = self.get_ee_pose()  # Get a geometry_msgs/PoseStamped msg
        self.ee_pos = np.array([ee_pos_tmp.pose.position.x, ee_pos_tmp.pose.position.y, ee_pos_tmp.pose.position.z])

        # --- 2. Get EE orientation
        self.ee_ori = np.array([ee_pos_tmp.pose.orientation.x, ee_pos_tmp.pose.orientation.y,
                                ee_pos_tmp.pose.orientation.z, ee_pos_tmp.pose.orientation.w])  # we need this for IK
        ee_ori_rpy = self.quaternion_to_euler(self.ee_ori)

        # --- Linear distance to the goal
        linear_dist_ee_goal = current_goal - self.cube_pos  # goal is box dtype and ee_pos is numpy.array. It is okay

        # --- 3. Vector to goal (we are giving only the direction vector)
        vec_ee_goal = self._safe_unit_vector(linear_dist_ee_goal)

        # --- 4. Euclidian distance
        euclidean_distance_cube_goal = scipy.spatial.distance.euclidean(self.cube_pos, current_goal)  # float

        # --- Get Current Joint values - only for the joints we are using
        #  we need this for delta actions
        # self.joint_values = self.current_joint_positions.copy()  # Get a float list
        self.joint_values = self.get_joint_angles()  # Get a float list
        # we don't need to convert this to numpy array since we concat using numpy below

        # --- 6. Get the previous action
        if self.prev_action is None:
            # Bootstrap: assume gripper is at max (open) on the first tick.
            # Once the agent's first action lands, self.prev_action is set
            # to the actual 7/4-DOF command in _set_action.
            if self.ee_action_type:
                prev_action = np.concatenate(
                    [self.ee_pos, [self.gripper_max]]).astype(np.float32)
            else:
                prev_action = np.concatenate(
                    [np.asarray(self.joint_values, dtype=np.float32),
                     [self.gripper_max]]).astype(np.float32)
        else:
            prev_action = self.prev_action.copy()

        # --- Get the joint velocities and joint positions using the joint_states topic
        if self.joint_pos_all is None or self.current_joint_velocities is None:
            done = False
            while not done:
                done = self._check_joint_states_ready()

        # Cube velocity via finite-diff. dt is the wall-clock gap since the
        # previous sample_observation call (~ environment_loop period in
        # real-time mode, action_cycle_time in normal MDP mode). The first
        # tick has no baseline so velocity stays at zero.
        now = rospy.get_time()
        if (self.prev_cube_pos is not None and self.prev_cube_rpy is not None
                and self.prev_cube_time is not None):
            dt = now - self.prev_cube_time
            if dt > 1e-6:
                self.cube_linear_velocity = ((self.cube_pos - self.prev_cube_pos)
                                             / dt).astype(np.float32)
                # Angular velocity: wrap each rpy delta into (-pi, pi] before
                # dividing so a small rotation across the +-pi seam doesn't
                # spike (rare for a manipulated cube, but cheap to guard).
                drpy = np.asarray(self.cube_ori, dtype=np.float32) - self.prev_cube_rpy
                drpy = (drpy + np.pi) % (2.0 * np.pi) - np.pi
                self.cube_angular_velocity = (drpy / dt).astype(np.float32)
        self.prev_cube_pos = self.cube_pos.astype(np.float32).copy()
        self.prev_cube_rpy = np.asarray(self.cube_ori, dtype=np.float32).copy()
        self.prev_cube_time = now

        # Cube position relative to EE - explicit feature for the approach
        # phase. Derivable from cube_pos and ee_pos, but giving the agent
        # the difference directly speeds up learning (FetchPickAndPlace
        # convention).
        cube_rel_to_ee = (self.cube_pos - self.ee_pos).astype(np.float32)

        # is_grasped: derived binary signal computed from cube-EE proximity
        # AND finger-close position. Robust lookup by joint name so we
        # don't depend on a fixed joint_pos_all ordering. NED2's left
        # gripper joint is ``joint_base_to_mors_1``.
        # The mors joints are PRISMATIC in metres (URDF: ±0.01 m stroke)
        # — same units as RX200's prismatic fingers, just symmetric
        # around 0 with a tighter range. grasp_finger_thresh in the
        # YAML is the position (m) below which we treat fingers as
        # "closed on something". TODO: empirically calibrate.
        try:
            _lf_idx = self.joint_state.name.index("joint_base_to_mors_1")
            _left_finger_pos = float(self.joint_state.position[_lf_idx])
        except (ValueError, AttributeError, IndexError):
            _left_finger_pos = self.gripper_max  # default: open
        _dist_ee_cube = float(np.linalg.norm(cube_rel_to_ee))
        self.is_grasped = float(
            (_dist_ee_cube < self.grasp_dist_thresh)
            and (_left_finger_pos < self.grasp_finger_thresh)
        )

        # Multi-goal phase update: once the cube has reached the lifted
        # intermediate goal (within reach_tolerance), flip the latch so
        # future obs/reward use the final pnp_goal.
        if (self.multi_goal and not self.intermediate_reached
                and self.intermediate_goal is not None):
            if self.check_if_reach_done(self.cube_pos, self.intermediate_goal):
                self.intermediate_reached = True

        # our observations
        obs = np.concatenate((self.ee_pos, ee_ori_rpy, vec_ee_goal, euclidean_distance_cube_goal,
                              self.joint_pos_all, prev_action, self.current_joint_velocities,
                              self.cube_pos, self.cube_ori,
                              self.cube_linear_velocity, self.cube_angular_velocity, cube_rel_to_ee,
                              [self.is_grasped]),
                             axis=None, dtype=np.float32)

        if self.log_internal_state:
            rospy.loginfo(f"Observations --->: {obs}")

        # GazeboGoalEnv.step() handles the outer
        # ``{observation, achieved_goal, desired_goal}`` Dict assembly via
        # separate ``_get_achieved_goal`` / ``_get_desired_goal`` hooks
        # (see below). sample_observation only returns the ``observation``
        # slot — either the flat array (normal_obs) or a nested sensor dict.
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

    def _get_achieved_goal(self):
        """Achieved goal = current cube position (HER tracks this).

        3-D (cube xyz) — matches the RX200 PnP goal template. is_grasped
        intentionally rides in the ``observation`` slot only, NOT here,
        so HER's cube-position relabeling stays well-defined.
        """
        if self.cube_pos is None:
            return np.zeros(3, dtype=np.float32)
        return self.cube_pos.astype(np.float32).copy()

    def _get_desired_goal(self):
        """Active desired goal for this timestep.

        Under multi_goal, exposes the intermediate (lift) goal until the
        cube reaches it, then the final pnp_goal. The agent's policy
        observes whatever this returns at step time; HER replays per
        timestep, so policies conditioned on the exposed goal stay
        consistent across relabeled samples.
        """
        if (self.multi_goal and not self.intermediate_reached
                and self.intermediate_goal is not None):
            return np.asarray(self.intermediate_goal, dtype=np.float32).copy()
        return np.asarray(self.pnp_goal, dtype=np.float32).copy()

    def compute_reward(self, achieved_goal, desired_goal, info) -> float:
        """
        Gymnasium GoalEnv HER hook. SB3's HERReplayBuffer passes a list-of-
        info-dicts (one per relabeled transition); my custom HER passes a
        single dict with ``is_her=True``. Dispatch per (ag, dg) pair so the
        relabeled rewards stay HER-safe (no step side-effects).
        """
        is_her = False
        if info is not None and isinstance(info, list):
            is_her = True
        elif info is not None and isinstance(info, dict):
            is_her = info.get('is_her', False)

        if is_her:
            info_tmp = {"is_her": True}
            rewards = [self.calculate_reward(ag, dg, info_tmp)
                       for ag, dg in zip(achieved_goal, desired_goal)]
            return np.array(rewards, dtype=np.float32)

        # Non-HER path: prefer cached reward_r (real-time loop) if available,
        # else compute fresh.
        if self.reward_r is not None and not is_her:
            return self.reward_r
        return self.calculate_reward(achieved_goal, desired_goal, info)

    def compute_terminated(self, achieved_goal, desired_goal, info):
        """
        Gymnasium GoalEnv hook. Pure function of (ag, dg) so HER relabeling
        agrees with the live env's termination check.
        """
        terminated = self.terminated_r
        self.info = self.info_r

        if "is_success" not in self.info:
            self.info["is_success"] = bool(terminated)

        if terminated is None:
            terminated = self.check_if_reach_done(achieved_goal, desired_goal)
            self.info["is_success"] = bool(terminated)

        return terminated

    def compute_truncated(self, achieved_goal, desired_goal, info):
        """GoalEnv hook — truncation is wrapper-driven (max_episode_steps)."""
        return self.truncated_r

    def calculate_reward(self, achieved_goal=None, desired_goal=None, info=None) -> float:
        """
        Reward for moving the cube toward the goal.

        Sparse: +1 on success, -1 otherwise. Pure function of (ag, dg).
        Dense:
            * simple (default and HER-safe): -euclidean(ag, dg)
            * layered: + reached_goal_reward / - mult_dist_reward * dist
              / + step_reward / + joint_limits_reward / + not_within_goal
              / + none_exe_reward (HER-INCOMPATIBLE — reads step-side-
              effect state; only used on the live env, never with HER)

        When called from compute_reward with explicit (ag, dg), uses those.
        Falls back to live env state (self.cube_pos, self.pnp_goal) for
        the in-loop call from the rospy.Timer-driven env_loop.
        """
        is_her = False
        if info is not None and isinstance(info, dict):
            is_her = info.get('is_her', False)

        if desired_goal is None:
            desired_goal = self.pnp_goal
        if achieved_goal is None:
            achieved_goal = self.cube_pos if self.cube_pos is not None else np.zeros(3, dtype=np.float32)

        reward = 0.0

        # Sparse — HER-safe.
        if self.reward_arc == "Sparse":
            reach_done = self.check_if_reach_done(achieved_goal, desired_goal)
            reward = 1.0 if reach_done else -1.0

            if not is_her:
                self.goal_marker.set_color(r=0.0, g=1.0) if reach_done else self.goal_marker.set_color(r=1.0, g=0.0)
                self.goal_marker.set_duration(duration=5)
                self.goal_marker.publish()
                if self.log_internal_state:
                    rospy.logwarn(">>>REWARD>>>" + str(reward))

        # Dense.
        else:
            # HER always uses simple dense (pure -distance) regardless of
            # simple_dense_reward, because the layered path reads step state
            # which doesn't survive HER's trajectory relabeling.
            if is_her:
                dist2goal = scipy.spatial.distance.euclidean(achieved_goal, desired_goal)
                reward += - dist2goal

            else:
                if self.simple_dense_reward:
                    dist2goal = scipy.spatial.distance.euclidean(achieved_goal, desired_goal)
                    reward += - dist2goal
                else:
                    done = self.check_if_reach_done(achieved_goal, desired_goal)
                    if done:
                        reward += self.reached_goal_reward
                        self.goal_marker.set_color(r=0.0, g=1.0)
                        self.goal_marker.set_duration(duration=5)
                    else:
                        self.goal_marker.set_color(r=1.0, g=0.0)
                        self.goal_marker.set_duration(duration=5)

                        # Grasp-aware layered dense shaping (PnP-specific).
                        # Pre-grasp: weight EE→cube distance (encourage
                        # approach). Post-grasp: weight cube→goal distance
                        # + grasp bonus. Without staging the agent gets the
                        # same cube→goal signal whether or not it's
                        # actually holding the cube, slowing credit
                        # assignment. Layered path is non-HER only; HER
                        # took the early `simple -distance` branch above.
                        dist_cube_goal = scipy.spatial.distance.euclidean(achieved_goal, desired_goal)
                        if self.is_grasped >= 0.5:
                            reward += - self.mult_dist_reward * dist_cube_goal
                            reward += 5.0  # grasp bonus
                        else:
                            dist_ee_cube = float(np.linalg.norm(
                                np.asarray(self.cube_pos, dtype=np.float32)
                                - np.asarray(self.ee_pos, dtype=np.float32)))
                            # Half-weight approach term so the post-grasp term
                            # dominates once the agent is holding the cube.
                            reward += - 0.5 * self.mult_dist_reward * dist_ee_cube

                        reward += self.step_reward

                    self.goal_marker.publish()

                    # Layered penalties — only safe on the live env.
                    reward += self.action_not_in_limits * self.joint_limits_reward
                    reward += (not self.within_goal_space) * self.not_within_goal_space_reward
                    if not self.movement_result:
                        reward += self.none_exe_reward

                if self.log_internal_state:
                    rospy.logwarn(">>>REWARD>>>" + str(reward))

        return reward

    def check_if_done(self):
        """
        Function to check if the episode is done.

        The Task is done if the Cube is close enough to the goal

        Returns:
            A boolean value indicating whether the episode has ended
            (e.g., because a goal has been reached or a failure condition has been triggered)
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

        # - Check if the Cube reached the goal (always against the FINAL
        # pnp_goal, never the intermediate lift target).
        done_reach = self.check_if_reach_done(self.cube_pos, self.pnp_goal)

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

    def quaternion_to_euler(self, quaternion):
        """
        Function to convert a quaternion to euler angles

        args:
            quaternion: a list of 4 elements representing a quaternion
        """
        # convert the quaternion to a rotation matrix
        rot_matrix = tf.transformations.quaternion_matrix(quaternion)

        # get the euler angles
        euler = tf.transformations.euler_from_matrix(rot_matrix)

        return euler

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
            goal = self._sample_box(self.goal_space)
            goal[2] = 0.015  # since the robot is mounted on a table

            if self.test_goal_pos(goal):
                return True, goal

        if self.log_internal_state:
            rospy.logdebug("Getting a random goal failed!")

        return False, None

    def get_random_goal_no_check(self):
        """
        Function to get a random goal without checking
        """
        random_goal = self._sample_box(self.goal_space)
        random_goal[2] = 0.015

        return random_goal

    def get_random_cube_init_pose(self):
        """
        Function to get a random cube pose for the initial position without checking

        return: random_cube_pose
        """
        random_cube_pose = self._sample_box(self.goal_space)
        random_cube_pose[2] = 0.015

        return random_cube_pose

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

        if self.debug:
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

        # Gripper scalar bounds — single value that the agent commands; we
        # discretize at the midpoint inside execute_action because NED2's
        # tools-commander API only accepts "open" / "close".
        self.gripper_min = float(rospy.get_param('/ned2/gripper_min'))
        self.gripper_max = float(rospy.get_param('/ned2/gripper_max'))

        # Grasp-detection thresholds (drive is_grasped in obs + dense reward)
        # and multi-goal lift height. NED2's left finger joint is the
        # ``joint_base_to_mors_1`` PRISMATIC joint (metres, URDF: ±0.01 m
        # stroke). Same units as RX200's prismatic fingers — only the
        # stroke magnitude differs, so the YAML threshold should be
        # tuned for NED2's tighter range.
        self.grasp_dist_thresh = float(rospy.get_param('/ned2/grasp_dist_thresh'))
        self.grasp_finger_thresh = float(rospy.get_param('/ned2/grasp_finger_thresh'))
        self.lift_height = float(rospy.get_param('/ned2/lift_height'))

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
        self.position_cube_min = rospy.get_param('/ned2/position_cube_min')
        self.position_cube_max = rospy.get_param('/ned2/position_cube_max')
        self.rpy_cube_min = rospy.get_param('/ned2/rpy_cube_min')
        self.rpy_cube_max = rospy.get_param('/ned2/rpy_cube_max')
        # Cube velocity bounds (finite-diff from cube pose) +
        # cube-relative-to-EE position. Added to bring obs closer to
        # FetchPickAndPlace SOTA — cube velocities help with sliding /
        # rolling / lift dynamics, rel-to-EE helps with the approach +
        # grasp phase.
        self.linear_velocity_cube_min = rospy.get_param('/ned2/linear_velocity_cube_min')
        self.linear_velocity_cube_max = rospy.get_param('/ned2/linear_velocity_cube_max')
        self.angular_velocity_cube_min = rospy.get_param('/ned2/angular_velocity_cube_min')
        self.angular_velocity_cube_max = rospy.get_param('/ned2/angular_velocity_cube_max')
        self.cube_rel_to_ee_min = rospy.get_param('/ned2/cube_rel_to_ee_min')
        self.cube_rel_to_ee_max = rospy.get_param('/ned2/cube_rel_to_ee_max')

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
        # TODO: confirm NED2 pnp world file. ned2_reach uses
        # 'ned2_workspace_only.world' via the train script; the call below
        # delegates to gazebo_core.launch_gazebo's defaults to mirror the
        # NED2 push goal template — the train script can override
        # custom_world_name='ned2_workspace_only.world' explicitly.
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
