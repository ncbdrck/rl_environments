#!/bin/python3

from gymnasium import spaces
from gymnasium.envs.registration import register
import numpy as np

from multiros.envs import GazeboBaseEnv

import rospy
import rostopic
from sensor_msgs.msg import JointState, PointCloud2, Image
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from cv_bridge import CvBridge
import cv2

# core modules of the framework
from multiros.utils import gazebo_core
from multiros.utils import gazebo_models
from multiros.utils import gazebo_physics
from multiros.utils.moveit_multiros import MoveitMultiros
from multiros.utils import ros_common
from multiros.utils import ros_controllers
from multiros.utils import ros_kinematics

from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_kinematics import KDLKinematics
from tf.transformations import euler_from_matrix, euler_from_quaternion

"""
Although it is best to register only the task environment, one can also register the robot environment. 
This is not necessary, but we can see if this section 
(Load the robot to gazebo and can control the robot with moveit or ros controllers)
works by calling "gymnasium.make" this env.
but you need to
    1. run gazebo - gazebo_core.launch_gazebo(launch_roscore=False, paused=False, pub_clock_frequency=100, gui=True)
    2. init a node - rospy.init_node('test_MyRobotGoalEnv')
    3. gymnasium.make("VX300SRobotEnv-v0")
"""
register(
    id='VX300SRobotEnv-v0',
    entry_point='rl_environments.vx300s.sim.robot_envs.vx300s_robot_sim:VX300SRobotEnv',
    max_episode_steps=1000,
)


class VX300SRobotEnv(GazeboBaseEnv.GazeboBaseEnv):
    """
    Superclass for all VX300S Robot environments.
    """

    def __init__(self, ros_port: str = None, gazebo_port: str = None, gazebo_pid=None, seed: int = None,
                 real_time: bool = False, action_cycle_time=0.0, load_cube: bool = False, load_table: bool = False,
                 use_kinect: bool = False):
        """
        Initializes a new Robot Environment

        Describe the robot and the sensors used in the env.

        Sensor Topic List:
            MoveIt: To get the pose and rpy of the robot.
            /joint_states: JointState received for the joints of the robot
            /head_mount_kinect2/depth/image_raw: Depth image from the kinect sensor
            /head_mount_kinect2/rgb/image_raw: RGB image from the kinect sensor

        Actuators Topic List:
            MoveIt: Send the joint positions to the robot.
            /vx300s/arm_controller/command: Send the joint positions to the robot.
            /vx300s/gripper_controller/command: Send the joint positions to the robot.
        """
        rospy.loginfo("Start Init VX300SRobotEnv Multiros")

        """
        Change the ros/gazebo master
        """
        if ros_port is not None:
            ros_common.change_ros_gazebo_master(ros_port=ros_port, gazebo_port=gazebo_port)

        """
        parameters
        """
        self.real_time = real_time  # if True, the simulation will run in real time

        # we don't need to pause/unpause gazebo if we are running in real time
        if self.real_time:
            unpause_pause_physics = False
        else:
            unpause_pause_physics = True

        """
        Unpause Gazebo
        """
        if not self.real_time:
            gazebo_core.unpause_gazebo()

        """
        Spawning the robot in Gazebo
        """
        spawn_robot = True

        # location of the robot URDF file
        urdf_pkg_name = "viperx300s_description"
        urdf_file_name = "vx300s_kinect.urdf.xacro"
        urdf_folder = "/urdf"

        # extra urdf args. Two upstream defaults need flipping:
        #   use_world_frame:=true      → declares <link name="world"/> so
        #                                 our kinect2 xacro can attach.
        #   load_gazebo_configs:=true  → pulls in interbotix_texture.gazebo
        #                                 which has the gazebo_ros_control
        #                                 plugin + Custom/Interbotix material.
        #                                 Without it the controller_manager
        #                                 never starts inside Gazebo.
        urdf_xacro_args = ["use_world_frame:=true", "load_gazebo_configs:=true"]

        # namespace of the robot
        namespace = "/vx300s"

        # robot state publisher
        robot_state_publisher_max_freq = None
        new_robot_state_term = False

        robot_model_name = "vx300s"
        robot_ref_frame = "world"

        # Set the initial pose of the robot model
        robot_pos_x = 0.0
        robot_pos_y = 0.0
        robot_pos_z = 0.0 if not load_table else 0.78
        robot_ori_w = 1.0
        robot_ori_x = 0.0
        robot_ori_y = 0.0
        robot_ori_z = 0.0

        # controller (must be inside above pkg_name/config/)
        controllers_file = "vx300s_controller.yaml"
        controllers_list = ["joint_state_controller", "arm_controller", "gripper_controller"]

        """
        Spawn other objects in Gazebo
        """
        # spawn a table
        self.load_table = load_table

        if load_table:
            gazebo_models.spawn_sdf_model_gazebo(pkg_name="viperx300s_description", file_name="model.sdf",
                                                 model_folder="/models/table",
                                                 model_name="table", namespace=namespace,
                                                 pos_x=0.2)

            # above function pauses the simulation, so we need to unpause it for real-time
            if self.real_time:
                gazebo_core.unpause_gazebo()

        # spawn a cube
        if load_cube:
            gazebo_models.spawn_sdf_model_gazebo(pkg_name="viperx300s_description", file_name="block.sdf",
                                                 model_folder="/models/block",
                                                 model_name="red_cube", namespace=namespace,
                                                 pos_x=0.35,
                                                 pos_z=0.795 if load_table else 0.015)

            # above function pauses the simulation, so we need to unpause it for real-time
            if self.real_time:
                gazebo_core.unpause_gazebo()

        """
        Set if the controllers in "controller_list" will be reset at the beginning of each episode, default is False.
        """
        reset_controllers = False

        """
        Set the reset mode of gazebo at the beginning of each episode
            "simulation": Reset gazebo simulation (Resets time) 
            "world": Reset Gazebo world (Does not reset time) - default

        resetting the "simulation" restarts the entire Gazebo environment, including all models and their positions, 
        while resetting the "world" retains the models but resets their properties and states within the world        
        """
        reset_mode = "world"

        """
        You can adjust the simulation step mode of Gazebo with two options:

            1. Using Unpause, set action and Pause gazebo
            2. Using the step function of Gazebo.

        By default, the simulation step mode is set to 1 (gazebo pause and unpause services). 
        However, if you choose simulation step mode 2, you can specify the number of steps Gazebo should take in each 
        iteration. The default value for this is 1.
        """
        sim_step_mode = 1
        num_gazebo_steps = 1

        """
        Set gazebo physics parameters to change the speed of the simulation
        """
        gazebo_max_update_rate = None
        gazebo_timestep = None

        if rospy.has_param('/vx300s/gazebo_update_rate_multiplier'):
            gazebo_max_update_rate = rospy.get_param('/vx300s/gazebo_update_rate_multiplier')
            rospy.loginfo(f"Applied Gazebo update_rate_multiplier = {gazebo_max_update_rate}")

        if rospy.has_param('/vx300s/gazebo_time_step'):
            gazebo_timestep = rospy.get_param('/vx300s/gazebo_time_step')
            rospy.loginfo(f"Applied Gazebo time_step = {gazebo_timestep}")

        """
        kill rosmaster at the end of the env
        """
        kill_rosmaster = True

        """
        kill gazebo at the end of the env
        """
        kill_gazebo = True

        """
        Clean ros Logs at the end of the env
        """
        clean_logs = False

        """
        Init GazeboBaseEnv.
        """
        super().__init__(
            spawn_robot=spawn_robot, urdf_pkg_name=urdf_pkg_name, urdf_file_name=urdf_file_name,
            urdf_folder=urdf_folder, urdf_xacro_args=urdf_xacro_args, namespace=namespace,
            robot_state_publisher_max_freq=robot_state_publisher_max_freq, new_robot_state_term=new_robot_state_term,
            robot_model_name=robot_model_name, robot_ref_frame=robot_ref_frame,
            robot_pos_x=robot_pos_x, robot_pos_y=robot_pos_y, robot_pos_z=robot_pos_z, robot_ori_w=robot_ori_w,
            robot_ori_x=robot_ori_x, robot_ori_y=robot_ori_y, robot_ori_z=robot_ori_z,
            controllers_file=controllers_file, controllers_list=controllers_list,
            reset_controllers=reset_controllers, reset_mode=reset_mode, sim_step_mode=sim_step_mode,
            num_gazebo_steps=num_gazebo_steps, gazebo_max_update_rate=gazebo_max_update_rate,
            gazebo_timestep=gazebo_timestep, kill_rosmaster=kill_rosmaster, kill_gazebo=kill_gazebo,
            clean_logs=clean_logs, ros_port=ros_port, gazebo_port=gazebo_port, gazebo_pid=gazebo_pid, seed=seed,
            unpause_pause_physics=unpause_pause_physics, action_cycle_time=action_cycle_time)

        """
        Define ros publisher, subscribers and services for robot and sensors
        """
        # ---------- joint state
        if namespace is not None and namespace != '/':
            self.joint_state_topic = namespace + "/joint_states"
        else:
            self.joint_state_topic = "/joint_states"

        self.joint_state_sub = rospy.Subscriber(self.joint_state_topic, JointState, self.joint_state_callback)
        self.joint_state = JointState()

        # ---------- Moveit
        ros_common.ros_launch_launcher(pkg_name="interbotix_xsarm_moveit_interface",
                                       launch_file_name="xsarm_moveit_interface.launch",
                                       args=["robot_model:=vx300s", "dof:=6", "use_python_interface:=true",
                                             "use_moveit_rviz:=false"])

        # ---------- kinect
        self.use_kinect = use_kinect

        if self.use_kinect:
            # depth image subscriber
            self.kinect_depth_sub = rospy.Subscriber("/head_mount_kinect2/depth/image_raw", Image,
                                                     self.kinect_depth_callback)
            self.kinect_depth = Image()
            self.cv_image_depth = None

            # rgb image subscriber
            self.kinect_rgb_sub = rospy.Subscriber("/head_mount_kinect2/rgb/image_raw", Image,
                                                   self.kinect_rgb_callback)
            self.kinect_rgb = Image()
            self.cv_image_rgb = None

        """
        Using the _check_connection_and_readiness method to check for the connection status of subscribers, publishers 
        and services
        """

        self._check_connection_and_readiness()

        """
        initialise controller and sensor objects here
        """
        self.arm_joint_names = ["waist",
                                "shoulder",
                                "elbow",
                                "forearm_roll",
                                "wrist_angle",
                                "wrist_rotate"]

        self.gripper_joint_names = ["left_finger",
                                    "right_finger"]

        if self.real_time:
            # we don't need to pause/unpause gazebo if we are running in real time
            self.move_VX300S_object = MoveitMultiros(arm_name='interbotix_arm',
                                                    gripper_name='interbotix_gripper',
                                                    robot_description="vx300s/robot_description",
                                                    ns="vx300s", pause_gazebo=False)
        else:
            self.move_VX300S_object = MoveitMultiros(arm_name='interbotix_arm',
                                                    gripper_name='interbotix_gripper',
                                                    robot_description="vx300s/robot_description",
                                                    ns="vx300s")

        # low-level control
        # rostopic for arm trajectory controller
        self.arm_controller_pub = rospy.Publisher('/vx300s/arm_controller/command',
                                                  JointTrajectory,
                                                  queue_size=10)

        # rostopic for gripper controller
        self.gripper_controller_pub = rospy.Publisher('/vx300s/gripper_controller/command',
                                                      JointTrajectory,
                                                      queue_size=10)

        # parameters for calculating FK, IK
        self.ee_link = "vx300s/ee_gripper_link"
        self.ref_frame = "vx300s/base_link"

        # Fk with pykdl_utils - old method
        self.pykdl_robot = URDF.from_parameter_server(key='vx300s/robot_description')
        self.kdl_kin = KDLKinematics(urdf=self.pykdl_robot, base_link=self.ref_frame, end_link=self.ee_link)

        # with ros_kinematics
        self.ros_kin = ros_kinematics.Kinematics_pyrobot(robot_description_parm="vx300s/robot_description",
                                                         base_link=self.ref_frame,
                                                         end_link=self.ee_link)

        # Per-link FK chains for the safety check in _check_action_links_safe.
        # Each subchain spans base_link → that link, so PyKDL's
        # ChainFkSolverPos_recursive expects len(q) == kin.num_joints (NOT
        # the full arm DOF). We store (kin, num_joints) so the safety check
        # can slice the action vector correctly per link. Building these
        # once at __init__ amortizes the URDF parse cost.
        self._safety_kin = {}
        for _link in self.SAFETY_CHECK_LINKS:
            try:
                _kin = KDLKinematics(urdf=self.pykdl_robot,
                                     base_link=self.ref_frame,
                                     end_link=_link)
                self._safety_kin[_link] = (_kin, int(_kin.num_joints))
            except Exception as _e:
                rospy.logwarn(f"[SAFETY] kinematics setup failed for {_link}: {_e}")

        """
        Finished __init__ method
        """
        if not self.real_time:
            gazebo_core.pause_gazebo()
        else:
            gazebo_core.unpause_gazebo()  # this is because loading models will pause the simulation
        rospy.loginfo("End Init VX300SRobotEnv")

    # ---------------------------------------------------
    #   Custom methods for the Custom Robot Environment

    """
    Define the custom methods for the environment
        * get_model_pose: Get the pose of an object in Gazebo
        * spawn_cube_in_gazebo: Spawn a cube in Gazebo
        * remove_cube_in_gazebo: Remove the cube from Gazebo
        * fk_pykdl: Function to calculate the forward kinematics of the robot arm. We are using pykdl_utils.
        * calculate_fk: Calculate the forward kinematics of the robot arm using the ros_kinematics package.
        * calculate_ik: Calculate the inverse kinematics of the robot arm using the ros_kinematics package.
        * joint_state_callback: Get the joint state of the robot
        * move_arm_joints: Set a joint position target only for the arm joints using low-level ros controllers.
        * move_gripper_joints: Set a joint position target only for the gripper joints using low-level ros controllers.
        * smooth_trajectory: Smooth the trajectory by interpolating between the current and target positions.
        * publish_trajectory: Publish the entire trajectory at once.
        * set_trajectory_joints: Set a joint position target only for the arm joints.
        * set_trajectory_ee: Set a pose target for the end effector of the robot arm.
        * get_ee_pose: Get end-effector pose a geometry_msgs/PoseStamped message
        * get_ee_rpy: Get end-effector orientation as a list of roll, pitch, and yaw angles.
        * get_joint_angles: Get current joint angles of the robot arm - 6 elements
        * check_goal: Check if the goal is reachable
        * check_goal_reachable_joint_pos: Check if the goal is reachable with joint positions
        * kinect_depth_callback: Callback function for kinect depth sensor
        * kinect_rgb_callback: Callback function for kinect rgb sensor
    """

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
                                                                            relative_entity_name="vx300s/base_link")

        if not self.real_time:
            gazebo_core.pause_gazebo()

        if success:
            if rpy:
                # The previous implementation passed an arbitrary 3x3 grid of
                # quaternion components to euler_from_matrix, which is NOT a
                # rotation matrix; the returned angles were meaningless.
                # Use the proper quaternion -> euler conversion instead.
                orientation = euler_from_quaternion(
                    [pose.orientation.x, pose.orientation.y,
                     pose.orientation.z, pose.orientation.w])

                orientation = np.array(orientation, dtype=np.float32)
            else:
                orientation = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z,
                                        pose.orientation.w], dtype=np.float32)

            position = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float32)

            # return position and orientation as numpy arrays
            return success, position, orientation

        # if the pose is not retrieved successfully
        return success, None, None

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
        done = gazebo_models.spawn_sdf_model_gazebo(pkg_name="viperx300s_description", file_name="block.sdf",
                                                    model_folder="/models/block",
                                                    model_name="red_cube", namespace="/vx300s",
                                                    pos_x=model_pos_x,
                                                    pos_y=model_pos_y,
                                                    pos_z=model_pos_z)

        # above function pauses the simulation, so we need to unpause it
        if self.real_time:
            gazebo_core.unpause_gazebo()

        return done

    def remove_cube_in_gazebo(self):
        """
        Remove the cube from Gazebo
        """
        done = gazebo_models.remove_model_gazebo(model_name="red_cube")

        # above function pauses the simulation, so we need to unpause it
        if self.real_time:
            gazebo_core.unpause_gazebo()

        return done

    def fk_pykdl(self, action):
        """
        Function to calculate the forward kinematics of the robot arm. We are using pykdl_utils.

        Args:
            action: joint positions of the robot arm (in radians)

        Returns:
            ee_position: end-effector position as a numpy array
        """
        # Calculate forward kinematics
        pose = self.kdl_kin.forward(action)

        # Extract position
        ee_position = np.array([pose[0, 3], pose[1, 3], pose[2, 3]], dtype=np.float32)
        # print("ee pos:", ee_position)  # for debugging
        # print("ee pos dtype:", type(ee_position))  # for debugging

        # Extract rotation matrix and convert to euler angles
        # ee_orientation = euler_from_matrix(pose[:3, :3], 'sxyz')

        return ee_position

    def calculate_fk(self, joint_positions, euler=True):
        """
        Calculate the forward kinematics of the robot arm using the ros_kinematics package.

        Args:
            joint_positions: joint positions of the robot arm (in radians)
            euler: True if the orientation is to be returned as euler angles (default: True)

        Returns:
            done: True if the FK calculation is successful
            ee_position: end-effector position as a numpy array
            ee_rpy: end-effector orientation as a list of rpy or quaternion values
        """
        done, ee_position, ee_ori = self.ros_kin.calculate_fk(joint_positions, des_frame=self.ee_link, euler=euler)

        return done, ee_position, ee_ori

    # Arm links whose world z must stay above the table for the action to be
    # safe. Order matches the URDF chain shoulder→ee_gripper. ee_arm_link /
    # gripper_prop_link / gripper_bar_link / fingers_link / left_finger_link
    # / right_finger_link are rigidly downstream of gripper_link, so
    # checking gripper_link covers them implicitly — saves ~4 FK calls per
    # tick without coverage loss.
    SAFETY_CHECK_LINKS = (
        "vx300s/shoulder_link",
        "vx300s/upper_arm_link",
        "vx300s/upper_forearm_link",
        "vx300s/lower_forearm_link",
        "vx300s/wrist_link",
        "vx300s/gripper_link",
        "vx300s/ee_gripper_link",
    )

    def _check_action_links_safe(self, joint_targets, current_joints=None):
        """
        Predict each arm link's world z under ``joint_targets`` and reject
        the action if any link would dip below ``table_z + safety_z_margin``.
        Also caps |target - current| per joint at ``max_joint_delta``.

        Uses the per-link ``KDLKinematics`` instances cached in
        ``self._safety_kin`` at __init__. Each subchain has its own joint
        count, so we slice ``q[:n]`` before calling ``forward`` — passing
        the full 5-DOF vector to a 1-joint subchain crashes the PyKDL C++
        extension (asked me how I know).

        Rosparams (all under ``/vx300s/``, with sim/real variants where the
        real value should be tighter):
          table_z, safety_z_margin[_real], max_joint_delta[_real]

        If ``current_joints`` is None, only the link-z check runs (the
        delta cap needs a baseline). Callers without current state can
        pass ``self.get_joint_angles()`` for a fresh read.

        Returns
        -------
        (safe, reason) : (bool, Optional[str])
            ``safe=True`` if every link is at or above the safety floor
            AND no joint exceeds the per-step delta cap. ``reason`` is a
            short string naming the first failure (link name + predicted
            z, OR joint index + delta).
        """
        strict = bool(getattr(self, "enable_strict_safety", False))
        table_z = float(rospy.get_param("/vx300s/table_z", -0.005))
        if strict:
            margin = float(rospy.get_param("/vx300s/safety_z_margin_strict", 0.030))
            max_delta = float(rospy.get_param("/vx300s/max_joint_delta_strict", 0.15))
        else:
            margin = float(rospy.get_param("/vx300s/safety_z_margin", 0.015))
            max_delta = float(rospy.get_param("/vx300s/max_joint_delta", 0.5))
        floor = table_z + margin

        q = np.asarray(joint_targets, dtype=np.float64)

        # Per-joint delta cap. Skipped when current pose is unknown
        # (bootstrap / first-tick calls).
        if current_joints is not None:
            cur = np.asarray(current_joints, dtype=np.float64)
            if cur.shape == q.shape:
                deltas = np.abs(q - cur)
                if np.any(deltas > max_delta):
                    idx = int(np.argmax(deltas))
                    return False, f"joint[{idx}] delta {deltas[idx]:.3f} > {max_delta}"

        # Per-link z check via cached pykdl subchains.
        per_link_z = []
        for link, (kin, n) in self._safety_kin.items():
            try:
                pose = kin.forward(q[:n])
            except Exception as e:
                # FK failure on a link means we don't know the geometry —
                # fail safe by rejecting the action.
                return False, f"FK failed for {link}: {e}"
            z = float(pose[2, 3])
            per_link_z.append((link, z))
            if z < floor:
                return False, f"{link} predicted z={z:.3f} < floor={floor:.3f}"

        # Debug: log the first few safety calls of the env's lifetime so we
        # can confirm the check is actually being entered (it should fire
        # once per env-loop tick once init_done=True and an action exists).
        # Counter is per-instance; not reset on reset() so the log stays
        # compact instead of repeating every episode.
        if not hasattr(self, "_safety_log_count"):
            self._safety_log_count = 0
        if self._safety_log_count < 3:
            self._safety_log_count += 1
            zs = ", ".join(f"{l.rsplit('/', 1)[-1]}={z:.3f}" for l, z in per_link_z)
            rospy.loginfo(f"[SAFETY] call #{self._safety_log_count}: floor={floor:.3f}, {zs}")

        return True, None

    def calculate_ik(self, target_pos, ee_ori=np.array([0.0, 0.0, 0.0, 1.0])):
        """
        Calculate the inverse kinematics of the robot arm using the ros_kinematics package.

        Args:
            target_pos: target end-effector position as a numpy array
            ee_ori: end-effector orientation as a list of quaternion values (default: [0.0, 0.0, 0.0, 1.0])

        Returns:
            done: True if the IK calculation is successful
            joint_positions: joint positions of the robot arm (in radians)
        """
        # define the pose in 1D array [x, y, z, qx, qy, qz, qw]
        target_pose = np.concatenate((target_pos, ee_ori))

        # get the current joint positions
        ee_position = self.get_joint_angles()

        done, joint_positions = self.ros_kin.calculate_ik(target_pose=target_pose, tolerance=[1e-3] * 6,
                                                          init_joint_positions=ee_position)

        return done, joint_positions

    def joint_state_callback(self, joint_state):
        """
        Function to get the joint state of the robot.
        """

        if joint_state is not None:
            self.joint_state = joint_state

            # joint names - not using this
            self.joint_state_names = list(joint_state.name)

            # get the current joint positions - using this
            joint_pos_all = list(joint_state.position)
            self.joint_pos_all = joint_pos_all

            # get the current joint velocities - we are using this
            self.current_joint_velocities = list(joint_state.velocity)

            # get the current joint efforts - not using this
            self.current_joint_efforts = list(joint_state.effort)

    def move_arm_joints(self, q_positions: np.ndarray, time_from_start: float = 0.5) -> bool:
        """
        Set a joint position target only for the arm joints using low-level ros controllers.

        Args:
            q_positions: joint positions of the robot arm
            time_from_start: time from start of the trajectory (set the speed to complete the trajectory)

        Returns:
            True if the action is successful
        """

        # create a JointTrajectory object
        trajectory = JointTrajectory()
        trajectory.joint_names = self.arm_joint_names
        trajectory.points.append(JointTrajectoryPoint())
        trajectory.points[0].positions = q_positions
        trajectory.points[0].velocities = [0.0] * len(self.arm_joint_names)
        trajectory.points[0].accelerations = [0.0] * len(self.arm_joint_names)
        trajectory.points[0].time_from_start = rospy.Duration(time_from_start)

        # send the trajectory to the controller
        self.arm_controller_pub.publish(trajectory)

        return True

    def move_gripper_joints(self, q_positions: np.ndarray, time_from_start: float = 0.5) -> bool:
        """
        Set a joint position target only for the gripper joints using low-level ros controllers.

        Args:
            q_positions: joint positions of the gripper
            time_from_start: time from start of the trajectory (set the speed to complete the trajectory)

        Returns:
            True if the action is successful
        """

        # create a JointTrajectory object
        trajectory = JointTrajectory()
        trajectory.joint_names = self.gripper_joint_names
        trajectory.points.append(JointTrajectoryPoint())
        trajectory.points[0].positions = q_positions
        trajectory.points[0].velocities = [0.0] * len(self.gripper_joint_names)
        trajectory.points[0].accelerations = [0.0] * len(self.gripper_joint_names)
        trajectory.points[0].time_from_start = rospy.Duration(time_from_start)

        # send the trajectory to the controller
        self.gripper_controller_pub.publish(trajectory)

        return True

    def smooth_trajectory(self, q_positions, time_from_start, multiplier=100):
        """
        Smooth the trajectory by interpolating between the current and target positions.

        Args:
            q_positions: target joint positions
            time_from_start: time from start of the trajectory (set the speed to complete the trajectory)
            multiplier: number of steps to interpolate between the current and target positions
        """
        num_steps = int(time_from_start * multiplier)  # Adjust the multiplier for more or fewer steps
        current_positions = self.joint_values
        delta_positions = (q_positions - current_positions) / num_steps

        trajectory_points = []
        for step in range(1, num_steps + 1):
            intermediate_positions = current_positions + step * delta_positions
            trajectory_points.append((intermediate_positions, time_from_start / num_steps * step))

        self.publish_trajectory(trajectory_points)

        return True

    def publish_trajectory(self, trajectory_points):
        """
        Publish the entire trajectory at once.

        Args:
            trajectory_points: List of tuples containing joint positions and time_from_start
        """
        trajectory = JointTrajectory()
        trajectory.joint_names = self.arm_joint_names

        for positions, time_from_start in trajectory_points:
            point = JointTrajectoryPoint()
            point.positions = positions
            point.velocities = [0.0] * len(self.arm_joint_names)
            point.accelerations = [0.0] * len(self.arm_joint_names)
            point.time_from_start = rospy.Duration(time_from_start)
            trajectory.points.append(point)

        # send the trajectory to the controller
        self.arm_controller_pub.publish(trajectory)

    def set_trajectory_joints(self, q_positions: np.ndarray) -> bool:
        """
        Set a joint position target only for the arm joints using moveit.
        """

        if self.real_time:
            # do not wait for the action to finish
            return self.move_VX300S_object.set_trajectory_joints(q_positions, async_move=True)
        else:
            return self.move_VX300S_object.set_trajectory_joints(q_positions)

    def set_trajectory_ee(self, pos: np.ndarray) -> bool:
        """
        Set a pose target for the end effector of the robot arm using moveit.
        """
        if self.real_time:
            # do not wait for the action to finish
            return self.move_VX300S_object.set_trajectory_ee(position=pos, async_move=True)
        else:
            return self.move_VX300S_object.set_trajectory_ee(position=pos)

    def get_ee_pose(self):
        """
        Returns the end-effector pose as a geometry_msgs/PoseStamped message

        This gives us the best pose if we are using the moveit config of the ReactorX repo
        They are getting the pose with ee_gripper_link
        """
        return self.move_VX300S_object.get_robot_pose()

    def get_ee_rpy(self):
        """
        Returns the end-effector orientation as a list of roll, pitch, and yaw angles.
        """
        return self.move_VX300S_object.get_robot_rpy()

    def get_joint_angles(self):
        """
        get current joint angles of the robot arm - 6 elements
        Returns a list
        """
        return self.move_VX300S_object.get_joint_angles_robot_arm()

    def check_goal(self, goal):
        """
        Check if the goal is reachable
        """
        return self.move_VX300S_object.check_goal(goal)

    def check_goal_reachable_joint_pos(self, joint_pos):
        """
        Check if the goal is reachable with joint positions
        """
        return self.move_VX300S_object.check_goal_joint_pos(joint_pos)

    def kinect_depth_callback(self, data):
        """
        Callback function for kinect depth sensor
        """
        self.kinect_depth = data

        # Convert ROS image message to OpenCV format (32FC1)
        bridge = CvBridge()
        cv_image_depth = bridge.imgmsg_to_cv2(data, desired_encoding="32FC1")
        self.cv_image_depth = cv_image_depth
        # print("Shape of depth:", cv_image_depth.shape)  # for debugging
        # todo: for the CNN policy
        # (480, 640) - for pytorch, this needs to be converted to (1, 480, 640)

    def kinect_rgb_callback(self, img_msg):
        """
        Callback function for kinect rgb sensor
        """
        self.kinect_rgb = img_msg
        bridge = CvBridge()

        # Convert ROS image message to OpenCV format (BGR)
        cv_image_bgr = bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")

        # Convert from BGR to RGB (required for pytorch or tensorflow CNNs) - (480, 640, 3)
        self.cv_image_rgb = cv2.cvtColor(cv_image_bgr, cv2.COLOR_BGR2RGB)
        # print("Shape of rgb:", cv_image_rgb.shape)  # for debugging
        # todo: for the CNN policy
        # (480, 640, 3) - for pytorch, this needs to be converted to (3, 480, 640)

    # helper fn for _check_connection_and_readiness
    def _check_joint_states_ready(self):
        """
        Function to check if the joint states are received
        """
        if not self.real_time:
            gazebo_core.unpause_gazebo()  # Unpause Gazebo physics

        # Wait for the service to be available
        rospy.logdebug(rostopic.get_topic_type(self.joint_state_topic, blocking=True))

        return True

    # helper fn for _check_connection_and_readiness
    def _check_moveit_ready(self):
        """
        Function to check if moveit services are running
        """
        rospy.wait_for_service("/vx300s/move_group/trajectory_execution/set_parameters")
        rospy.logdebug(rostopic.get_topic_type("/vx300s/planning_scene", blocking=True))
        rospy.logdebug(rostopic.get_topic_type("/vx300s/move_group/status", blocking=True))

        return True

    # helper fn for _check_connection_and_readiness
    def _check_ros_controllers_ready(self):
        """
        Function to check if ros controllers are running
        """
        rospy.logdebug(rostopic.get_topic_type("/vx300s/arm_controller/state", blocking=True))
        rospy.logdebug(rostopic.get_topic_type("/vx300s/gripper_controller/state", blocking=True))

        return True

    def _check_kinect_ready(self):
        """
        Function to check if kinect sensor is running
        """
        rospy.logdebug(rostopic.get_topic_type("/head_mount_kinect2/depth/points", blocking=True))

        return True

    def _check_connection_and_readiness(self):
        """
        Function to check the connection status of subscribers, publishers and services, as well as the readiness of
        all systems.
        """
        self._check_moveit_ready()
        self._check_joint_states_ready()
        self._check_ros_controllers_ready()

        if self.use_kinect:
            self._check_kinect_ready()

        rospy.loginfo("All system are ready!")

        return True
