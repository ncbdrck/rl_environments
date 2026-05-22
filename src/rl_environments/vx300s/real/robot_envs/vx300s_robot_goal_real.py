#!/bin/python3

import rospy
import rostopic
from gymnasium.envs.registration import register

import numpy as np
from sensor_msgs.msg import JointState, PointCloud2, Image
from geometry_msgs.msg import Pose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

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
    id='VX300SRobotGoalEnv-v0',
    entry_point='rl_environments.vx300s.real.robot_envs.vx300s_robot_goal_real:VX300SRobotGoalEnv',
    max_episode_steps=1000,
)


class VX300SRobotGoalEnv(RealGoalEnv.RealGoalEnv):
    """
    Superclass for all real VX300S Robot environments. - For goal-conditioned tasks
    """

    def __init__(self, ros_port: str = None, seed: int = None, close_env_prompt: bool = False, action_cycle_time=0.0,
                 use_kinect: bool = False, use_zed2: bool = False):
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
            /vx300s/arm_controller/command: Send the joint positions to the robot.
            /vx300s/gripper_controller/command: Send the joint positions to the robot
        """
        rospy.loginfo("Start Init VX300SRobotGoalEnv RealROS")

        if ros_port is not None:
            ros_common.change_ros_master(ros_port=ros_port)

        # none for now

        load_robot = True
        robot_pkg_name = "interbotix_xsarm_moveit_interface"
        robot_launch_file = "xsarm_moveit_interface.launch"
        robot_args = ["robot_model:=vx300s", "use_actual:=true", "dof:=6", "use_python_interface:=true",
                      "use_moveit_rviz:=true"]

        namespace = "/vx300s"

        kill_rosmaster = False

        clean_logs = True

        super().__init__(
            load_robot=load_robot, robot_pkg_name=robot_pkg_name, robot_launch_file=robot_launch_file,
            robot_args=robot_args, namespace=namespace, kill_rosmaster=kill_rosmaster, clean_logs=clean_logs,
            ros_port=ros_port, seed=seed, close_env_prompt=close_env_prompt, action_cycle_time=action_cycle_time)

        # ---------- joint state
        if namespace is not None and namespace != '/':
            self.joint_state_topic = namespace + "/joint_states"
        else:
            self.joint_state_topic = "/joint_states"

        self.joint_state_sub = rospy.Subscriber(self.joint_state_topic, JointState, self.joint_state_callback)
        self.joint_state = JointState()

        #  Moveit object
        self.move_VX300S_object = MoveitRealROS(arm_name='interbotix_arm',
                                               gripper_name='interbotix_gripper',
                                               robot_description="vx300s/robot_description",
                                               ns="vx300s")

        # ---------- kinect or zed2
        self.use_kinect = use_kinect
        self.use_zed2 = use_zed2

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
            self.zed2_depth_sub = rospy.Subscriber("/vx300s/zed2/depth/depth_registered", Image,
                                                        self.zed2_depth_callback)
            self.zed2_depth = Image()
            self.cv_image_depth = None

            # rgb image subscriber
            self.zed2_rgb_sub = rospy.Subscriber("/vx300s/zed2/left/image_rect_color", Image,
                                                   self.zed2_rgb_callback)
            self.zed2_rgb = Image()
            self.cv_image_rgb = None

        self._check_connection_and_readiness()

        # For ROS Controllers
        self.arm_joint_names = ["waist",
                                "shoulder",
                                "elbow",
                                "forearm_roll",
                                "wrist_angle",
                                "wrist_rotate"]

        self.gripper_joint_names = ["left_finger",
                                    "right_finger"]

        # low-level control
        # The rostopic for joint trajectory controller
        self.arm_controller_pub = rospy.Publisher('/vx300s/arm_controller/command',
                                                               JointTrajectory,
                                                               queue_size=10)

        # rostopic for gripper controller
        self.gripper_controller_pub = rospy.Publisher('/vx300s/gripper_controller/command',
                                                      JointTrajectory,
                                                      queue_size=10)

        # parameters for calculating FK
        self.ee_link = "vx300s/ee_gripper_link"
        self.ref_frame = "vx300s/base_link"

        # Fk with pykdl_utils
        self.pykdl_robot = URDF.from_parameter_server(key='vx300s/robot_description')
        self.kdl_kin = KDLKinematics(urdf=self.pykdl_robot, base_link=self.ref_frame, end_link=self.ee_link)

        # with ros_kinematics
        self.ros_kin = ros_kinematics.Kinematics_pyrobot(robot_description_parm="vx300s/robot_description",
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

        # Per-link FK chains for the safety check. Each subchain spans
        # base_link → that link, so PyKDL expects len(q) == kin.num_joints
        # (NOT the full arm DOF). Cached at __init__ to amortize the URDF
        # parse cost.
        self._safety_kin = {}
        for _link in self.SAFETY_CHECK_LINKS:
            try:
                _kin = KDLKinematics(urdf=self.pykdl_robot,
                                     base_link=self.ref_frame,
                                     end_link=_link)
                self._safety_kin[_link] = (_kin, int(_kin.num_joints))
            except Exception as _e:
                rospy.logwarn(f"[SAFETY] kinematics setup failed for {_link}: {_e}")

        rospy.loginfo("End Init VX300SRobotGoalEnv")

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

        Also stamps ``_latest_joint_state_time`` so the env_loop can
        gate ticks on driver freshness (real-side only).
        """

        if joint_state is not None:
            self.joint_state = joint_state
            self._latest_joint_state_time = rospy.get_time()

            # joint names - not using this
            self.joint_state_names = list(joint_state.name)

            # get the current joint positions - using this
            joint_pos_all = list(joint_state.position)
            self.joint_pos_all = joint_pos_all

            # get the current joint velocities - we are using this
            self.current_joint_velocities = list(joint_state.velocity)

            # get the current joint efforts - not using this
            self.current_joint_efforts = list(joint_state.effort)

    # Arm links whose world z must stay above the table for the action to be
    # safe. Order matches the URDF chain shoulder → ee_gripper. Links rigidly
    # downstream of gripper_link are covered implicitly.
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
        table_z = float(rospy.get_param("/vx300s/table_z", -0.005))
        if strict:
            margin = float(rospy.get_param("/vx300s/safety_z_margin_strict", 0.030))
            max_delta = float(rospy.get_param("/vx300s/max_joint_delta_strict", 0.15))
        else:
            margin = float(rospy.get_param("/vx300s/safety_z_margin", 0.015))
            max_delta = float(rospy.get_param("/vx300s/max_joint_delta", 0.5))
        floor = table_z + margin

        q = np.asarray(joint_targets, dtype=np.float64)

        if current_joints is not None:
            cur = np.asarray(current_joints, dtype=np.float64)
            if cur.shape == q.shape:
                deltas = np.abs(q - cur)
                if np.any(deltas > max_delta):
                    idx = int(np.argmax(deltas))
                    return False, f"joint[{idx}] delta {deltas[idx]:.3f} > {max_delta}"

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
            rospy.loginfo(f"[SAFETY-REAL] call #{self._safety_log_count}: floor={floor:.3f}, {zs}")

        return True, None

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
        Set a joint position target only for the arm joints.
        """
        return self.move_VX300S_object.set_trajectory_joints(q_positions, async_move=True)

    def set_trajectory_ee(self, pos: np.ndarray) -> bool:
        """
        Set a pose target for the end effector of the robot arm.
        """
        return self.move_VX300S_object.set_trajectory_ee(position=pos, async_move=True)

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

    def _check_zed2_ready(self):
        """
        Function to check if zed2 sensor is running
        """
        rospy.logdebug(rostopic.get_topic_type("/vx300s/zed2/left/image_rect_color", blocking=True))

        return True

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