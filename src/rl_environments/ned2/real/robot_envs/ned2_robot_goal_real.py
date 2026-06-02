#!/bin/python3

import rospy
import rostopic
from gymnasium.envs.registration import register

import numpy as np
from sensor_msgs.msg import JointState, PointCloud2, Image, CompressedImage
from geometry_msgs.msg import Pose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from niryo_robot_tools_commander.msg import ToolAction, ToolGoal, ToolCommand
import actionlib

# core modules of the framework
from realros.utils.moveit_realros import MoveitRealROS
from realros.utils import ros_common
from realros.utils import ros_controllers
from realros.utils import ros_markers
from realros.utils import ros_kinematics

from realros.envs import RealGoalEnv

from cv_bridge import CvBridge
import cv2

from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_kinematics import KDLKinematics
from tf.transformations import euler_from_matrix

register(
    id='NED2RobotGoalEnv-v0',
    entry_point='rl_environments.ned2.real.robot_envs.ned2_robot_goal_real:NED2RobotGoalEnv',
    max_episode_steps=1000,
)


class NED2RobotGoalEnv(RealGoalEnv.RealGoalEnv):
    """
    Superclass for all real NED2 Robot environments. - For goal-conditioned tasks
    """

    def __init__(self, ros_port: str = None, seed: int = None, close_env_prompt: bool = False, action_cycle_time=0.0,
                 use_kinect: bool = False, use_zed2: bool = False, use_wrist_camera: bool = False,
                 remote_ip: str = None, local_ip:str = None, multi_device_mode: bool = True):
        """
        Initializes a new Robot Goal Environment

        Describe the robot and the sensors used in the env.

        Sensor Topic List:
            MoveIt: To get the pose and rpy of the robot.
            /joint_states: JointState received for the joints of the robot
            /head_mount_kinect2/depth/image_raw: Depth image from the kinect sensor
            /head_mount_kinect2/rgb/image_raw: RGB image from the kinect sensor

        Actuators Topic List:
            MoveIt!: Send the joint positions to the robot.
            /niryo_robot_follow_joint_trajectory_controller/command: Send the joint positions to the robot.
            /niryo_robot_tools_commander/action_server: Send the joint positions to the robot gripper.
        """
        rospy.loginfo("Start Init NED2RobotGoalEnv RealROS")

        if multi_device_mode:
            ros_common.change_ros_master_multi_device(remote_ip=remote_ip,
                                                      local_ip=local_ip, remote_ros_port=ros_port)

        elif ros_port is not None:
            ros_common.change_ros_master(ros_port=ros_port)

        # none for now

        load_robot = False

        kill_rosmaster = False

        clean_logs = True

        super().__init__(
            load_robot=load_robot, kill_rosmaster=kill_rosmaster, clean_logs=clean_logs,
            ros_port=ros_port, seed=seed, close_env_prompt=close_env_prompt, action_cycle_time=action_cycle_time,
            multi_device_mode=multi_device_mode, remote_ip=remote_ip, local_ip=local_ip)
        # ---------- joint state
        self.joint_state_topic = "/joint_states"

        self.joint_state_sub = rospy.Subscriber(self.joint_state_topic, JointState, self.joint_state_callback)
        self.joint_state = JointState()

        # Moveit object
        self.move_NED2_object = MoveitRealROS(arm_name='arm',
                                               robot_description="/robot_description",
                                               ns="/")

        # ---------- kinect / zed2 / Niryo wrist camera
        self.use_kinect = use_kinect
        self.use_zed2 = use_zed2
        self.use_wrist_camera = use_wrist_camera

        # todo: find the actual topic names
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

        # todo: find the actual topic names
        if self.use_zed2:
            # depth image subscriber
            self.zed2_depth_sub = rospy.Subscriber("/zed2/depth/depth_registered", Image,
                                                        self.zed2_depth_callback)
            self.zed2_depth = Image()
            self.cv_image_depth = None

            # rgb image subscriber
            self.zed2_rgb_sub = rospy.Subscriber("/zed2/left/image_rect_color", Image,
                                                   self.zed2_rgb_callback)
            self.zed2_rgb = Image()
            self.cv_image_rgb = None

        # Niryo built-in wrist camera (real). See ned2_robot_real.py
        # for the full rationale — Niryo publishes only a compressed
        # stream via the niryo_robot_vision node.
        if self.use_wrist_camera:
            self.wrist_camera_rgb_sub = rospy.Subscriber(
                "/niryo_robot_vision/compressed_video_stream",
                CompressedImage,
                self.wrist_camera_rgb_callback,
                queue_size=1,
            )
            self.wrist_camera_rgb = CompressedImage()
            self.cv_image_wrist = None

        self._check_connection_and_readiness()

        # For ROS Controllers
        self.arm_joint_names = ["joint_1",
                                "joint_2",
                                "joint_3",
                                "joint_4",
                                "joint_5",
                                "joint_6"]

        # Physical gripper joints (mors_1 / mors_2 prismatic fingers) — the
        # joint_state_callback references this list to build the obs-facing
        # joint vector by name.
        self.gripper_joint_names = ["joint_base_to_mors_1",
                                    "joint_base_to_mors_2"]

        # low-level control
        # The rostopic for joint trajectory controller
        self.arm_controller_pub = rospy.Publisher('niryo_robot_follow_joint_trajectory_controller/command',
                                                               JointTrajectory,
                                                               queue_size=10)

        # parameters for calculating FK, IK
        # tool_link, not wrist_link: the action vector has 6 joint values
        # (NED2 is 6-DOF), but base_link -> wrist_link is a 5-joint chain
        # (joint_1..joint_5). KDL needs len(q) == subchain joints, so an
        # ee_link with 6 joints in the chain is required. tool_link is
        # also what Niryo's SRDF declares as the planning-group EE
        # (group "arm" ends at tool_link), so this aligns FK with the
        # MoveIt planning target.
        self.ee_link = "tool_link"
        self.ref_frame = "base_link"

        # Fk with pykdl_utils
        self.pykdl_robot = URDF.from_parameter_server(key='/robot_description')
        self.kdl_kin = KDLKinematics(urdf=self.pykdl_robot, base_link=self.ref_frame, end_link=self.ee_link)

        # with ros_kinematics
        self.ros_kin = ros_kinematics.Kinematics_pyrobot(robot_description_parm="/robot_description",
                                                         base_link=self.ref_frame,
                                                         end_link=self.ee_link)

        # Strict-safety flag — _check_action_links_safe picks up the
        # tighter margins (safety_z_margin_strict, max_joint_delta_strict)
        # when this is True. Defaults True on the real robot env.
        self.enable_strict_safety = True

        # Joint-state freshness tracker. The task env's env_loop gates
        # on (now - _latest_joint_state_time) so a dead driver / cable
        # disconnect doesn't keep publishing actions against frozen
        # state.
        self._latest_joint_state_time = None

        # Per-link FK chains for the safety check. KDLKinematics needs
        # one solver per subchain; tool_link only exists when the
        # gripper URDF is loaded. Missing links are skipped at build
        # with a warning. See RX200RobotEnv (d8b5517).
        self._safety_kin = {}
        for _link in self.SAFETY_CHECK_LINKS:
            rospy.loginfo(f"[SAFETY] building kinematics for {_link} ...")
            try:
                _kin = KDLKinematics(urdf=self.pykdl_robot,
                                     base_link=self.ref_frame,
                                     end_link=_link)
                self._safety_kin[_link] = (_kin, int(_kin.num_joints))
                rospy.loginfo(f"[SAFETY] {_link} ok ({_kin.num_joints} joints)")
            except Exception as _e:
                rospy.logwarn(f"[SAFETY] kinematics setup failed for {_link}: {_e}")

        rospy.loginfo("End Init NED2RobotGoalEnv")

    # ---------------------------------------------------
    #   Custom methods for the Robot Environment

    def fk_pykdl(self, action):
        """
        Function to calculate the forward kinematics of the robot arm. We are using pykdl_utils.

        Args:
            action: joint positions of the robot arm (in radians)

        Returns:
            ee_position: end-effector position as a numpy array
        """
        # Defensive: callers (sample_observation) hit this before the
        # first /joint_states callback can populate self.joint_pos_all
        # in some env-loop startup races. Return None so the caller's
        # if ee is None: ee = self.ee_pos fallback keeps the obs
        # in a defined state instead of raising from KDL.
        if action is None or len(action) == 0:
            return None

        # Calculate forward kinematics
        pose = self.kdl_kin.forward(action)

        # Extract position
        ee_position = np.array([pose[0, 3], pose[1, 3], pose[2, 3]], dtype=np.float32)  # we need to convert to float32

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

        Also stamps ``_latest_joint_state_time`` so task envs can
        detect driver / cable disconnects via env_loop freshness gates.
        """

        if joint_state is not None:
            self.joint_state = joint_state
            self._latest_joint_state_time = rospy.get_time()

            # joint names — used below to pull positions/velocities by
            # name rather than trusting the driver's publish order.
            self.joint_state_names = list(joint_state.name)

            # Build the obs-facing joint vectors by NAME lookup so a
            # driver change that re-orders /joint_states (or adds an
            # extra finger / mimic joint) doesn't silently scramble the
            # observation. The expected joint set is arm + (gripper
            # when the gripper URDF is loaded).
            #
            # Race guard: the joint_state subscriber is registered early
            # in __init__ (so the connection-readiness check can use
            # it), but ``arm_joint_names`` / ``gripper_joint_names`` are
            # populated later. On real hardware /joint_states may already
            # be publishing, so the callback can fire before those
            # attributes exist. Skip the build until they're set.
            if not hasattr(self, "arm_joint_names") or not hasattr(self, "gripper_joint_names"):
                return
            wanted = list(self.arm_joint_names) + list(self.gripper_joint_names)
            name_to_idx = {n: i for i, n in enumerate(joint_state.name)}
            indices = [name_to_idx[n] for n in wanted if n in name_to_idx]

            self.joint_pos_all = [joint_state.position[i] for i in indices]
            self.current_joint_velocities = [joint_state.velocity[i] for i in indices]
            self.current_joint_efforts = [joint_state.effort[i] for i in indices]
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

    def move_gripper_joints(self, action: str) -> bool:
        """
        Set a joint position target only for the gripper joints using low-level ros controllers. - ros action server

        Args:
            action: "open" or "close" the gripper

        Returns:
            True if the action is successful
        """

        # for the gripper
        gripper_action_client = actionlib.SimpleActionClient('/niryo_robot_tools_commander/action_server',
                                                                  ToolAction)

        # wait for the action server to start
        gripper_action_client.wait_for_server()

        # create a ToolGoal object
        tool_goal = ToolGoal()
        tool_goal.cmd.tool_id = 11  # gripper tool id (normal gripper)
        tool_goal.cmd.cmd_type = ToolCommand.OPEN_GRIPPER if action == "open" else ToolCommand.CLOSE_GRIPPER

        # send the goal to the action server
        gripper_action_client.send_goal(tool_goal)
        # wait for the action to complete
        gripper_action_client.wait_for_result()

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
        Set a joint position target only for the arm joints.
        """
        return self.move_NED2_object.set_trajectory_joints(q_positions, async_move=True)

    def set_trajectory_ee(self, pos: np.ndarray) -> bool:
        """
        Set a pose target for the end effector of the robot arm.
        """
        return self.move_NED2_object.set_trajectory_ee(position=pos, async_move=True)

    def get_ee_pose(self):
        """
        Returns the end-effector pose as a geometry_msgs/PoseStamped message
        """
        return self.move_NED2_object.get_robot_pose()

    def get_ee_rpy(self):
        """
        Returns the end-effector orientation as a list of roll, pitch, and yaw angles.
        """
        return self.move_NED2_object.get_robot_rpy()

    def get_joint_angles(self):
        """
        get current joint angles of the robot arm - 6 elements
        Returns a list
        """
        return self.move_NED2_object.get_joint_angles_robot_arm()

    def check_goal(self, goal):
        """
        Check if the goal is reachable
        """
        return self.move_NED2_object.check_goal(goal)

    def check_goal_reachable_joint_pos(self, joint_pos):
        """
        Check if the goal is reachable with joint positions
        """
        return self.move_NED2_object.check_goal_joint_pos(joint_pos)

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

    def wrist_camera_rgb_callback(self, img_msg):
        """
        Callback for Niryo's built-in wrist camera (real).
        Source: /niryo_robot_vision/compressed_video_stream
        (sensor_msgs/CompressedImage). cv_bridge decodes the compressed
        bytes; result goes to self.cv_image_wrist as RGB.
        """
        self.wrist_camera_rgb = img_msg
        bridge = CvBridge()
        cv_image_bgr = bridge.compressed_imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        self.cv_image_wrist = cv2.cvtColor(cv_image_bgr, cv2.COLOR_BGR2RGB)

    def zed2_depth_callback(self, data):
        """
        Callback function for zed2 depth sensor
        """
        self.zed2_depth = data

        # Convert ROS image message to OpenCV format (32FC1)
        bridge = CvBridge()
        cv_image_depth = bridge.imgmsg_to_cv2(data, desired_encoding="32FC1")
        self.cv_image_depth = cv_image_depth
        # print("Shape of depth:", cv_image_depth.shape)  # for debugging
        # todo: for the CNN policy
        # (720, 1280) - for pytorch, this needs to be converted to (1, 720, 1280)

    def zed2_rgb_callback(self, img_msg):
        """
        Callback function for zed2 rgb sensor
        """
        self.zed2_rgb = img_msg
        bridge = CvBridge()

        # Convert ROS image message to OpenCV format (BGR)
        cv_image_bgr = bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")

        # Convert from BGR to RGB (required for pytorch or tensorflow CNNs) - (720, 1280, 3)
        self.cv_image_rgb = cv2.cvtColor(cv_image_bgr, cv2.COLOR_BGR2RGB)
        # print("Shape of rgb:", cv_image_rgb.shape)  # for debugging
        # todo: for the CNN policy
        # (720, 1280, 3) - for pytorch, this needs to be converted to (3, 720, 1280)

    # helper fn for _check_connection_and_readiness
    def _check_joint_states_ready(self):
        """
        Function to check if the joint states are received
        """
        # Wait for the service to be available
        rospy.logdebug(rostopic.get_topic_type(self.joint_state_topic, blocking=True))

        return True

    def _check_moveit_ready(self):
        """
        Function to check if moveit services are running
        """
        rospy.wait_for_service("/move_group/trajectory_execution/set_parameters")
        rospy.logdebug(rostopic.get_topic_type("/planning_scene", blocking=True))
        rospy.logdebug(rostopic.get_topic_type("/move_group/status", blocking=True))

        return True

    # helper fn for _check_connection_and_readiness
    def _check_ros_controllers_ready(self):
        """
        Function to check if ros controllers are running
        """
        rospy.logdebug(rostopic.get_topic_type("/niryo_robot_follow_joint_trajectory_controller/state", blocking=True))

        return True

    def _check_kinect_ready(self):
        """
        Function to check if kinect sensor is running
        """
        rospy.logdebug(rostopic.get_topic_type("/head_mount_kinect2/depth/points", blocking=True))

        return True

    def _check_zed2_ready(self):
        """
        Function to check if zed2 sensor is running
        """
        rospy.logdebug(rostopic.get_topic_type("/zed2/left/image_rect_color", blocking=True))

        return True

    # ---------------------------------------------------
    #   Per-link FK safety (mirrors RX200RobotGoalEnv for Ned2 real)

    # Arm links whose world z must stay above the table/ground for the
    # action to be safe. Order matches URDF chain shoulder→tool.
    # tool_link only exists if the gripper URDF is loaded (PnP runs);
    # missing links are skipped at build time with a warning.
    SAFETY_CHECK_LINKS = (
        "shoulder_link",
        "arm_link",
        "elbow_link",
        "forearm_link",
        "wrist_link",
        "tool_link",
    )

    def _check_action_links_safe(self, joint_targets, current_joints=None):
        """Predict each arm link's world z under ``joint_targets`` and reject
        the action if any link would dip below ``table_z + safety_z_margin``.
        Also caps |target - current| per joint at ``max_joint_delta``.

        When ``self.enable_strict_safety`` is True (default on the real
        robot env) the tighter margins are used:
          safety_z_margin_strict (default 0.030)
          max_joint_delta_strict (default 0.15)

        Returns
        -------
        (safe, reason) : (bool, Optional[str])
        """
        strict = bool(getattr(self, "enable_strict_safety", False))
        table_z = float(rospy.get_param("/ned2/table_z", -0.005))
        if strict:
            margin = float(rospy.get_param("/ned2/safety_z_margin_strict", 0.030))
            max_delta = float(rospy.get_param("/ned2/max_joint_delta_strict", 0.15))
        else:
            margin = float(rospy.get_param("/ned2/safety_z_margin", 0.015))
            max_delta = float(rospy.get_param("/ned2/max_joint_delta", 0.5))
        floor = table_z + margin

        q = np.asarray(joint_targets, dtype=np.float64)

        # Per-joint delta cap (skipped when current pose is unknown).
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
                return False, f"FK failed for {link}: {e}"
            z = float(pose[2, 3])
            per_link_z.append((link, z))
            if z < floor:
                return False, f"{link} predicted z={z:.3f} < floor={floor:.3f}"

        if not hasattr(self, "_safety_log_count"):
            self._safety_log_count = 0
        if self._safety_log_count < 3:
            self._safety_log_count += 1
            zs = ", ".join(f"{l.rsplit('/', 1)[-1]}={z:.3f}" for l, z in per_link_z)
            rospy.loginfo(f"[SAFETY] call #{self._safety_log_count}: floor={floor:.3f}, {zs}")

        return True, None

    # ---------------------------------------------------
    #   Methods to override in the Robot Environment

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
        if self.use_zed2:
            self._check_zed2_ready()

        rospy.loginfo("All system are ready!")

        return True