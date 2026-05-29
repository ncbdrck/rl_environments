#!/bin/python3

from gymnasium import spaces
from gymnasium.envs.registration import register
import numpy as np

from multiros.envs import GazeboGoalEnv

import rospy
import rostopic
from sensor_msgs.msg import JointState, PointCloud2, Image
from geometry_msgs.msg import Pose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
import actionlib

from cv_bridge import CvBridge
import cv2

# core modules of the framework
from multiros.utils import gazebo_core
from multiros.utils import gazebo_models
from multiros.utils import gazebo_physics
from multiros.utils.moveit_multiros import MoveitMultiros
from multiros.utils import ros_common
from multiros.utils import ros_controllers
from multiros.utils import ros_markers
from multiros.utils import ros_kinematics

from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_kinematics import KDLKinematics
from tf.transformations import euler_from_matrix, euler_from_quaternion

register(
    id='NED2RobotGoalBaseSimEnv-v0',
    entry_point='rl_environments.ned2.sim.robot_envs.ned2_robot_goal_sim:NED2RobotGoalEnv',
    max_episode_steps=1000,
)


class NED2RobotGoalEnv(GazeboGoalEnv.GazeboGoalEnv):
    """
    Superclass for all NED2 Robot Goal environments.
    """

    def __init__(self, ros_port: str = None, gazebo_port: str = None, gazebo_pid=None, seed: int = None,
                 real_time: bool = False, action_cycle_time=0.0, load_cube: bool = False, load_table: bool = False,
                 use_camera: bool = False, use_wrist_camera: bool = False,
                 gripper: bool = False):
        """
        Initializes a new Robot Goal Environment

        Describe the robot and the sensors used in the env.

        Sensor Topic List:
            MoveIt: To get the pose and rpy of the robot.
            /joint_states: JointState received for the joints of the robot
            /gazebo_camera/image_raw: RGB image from the robot camera.


        Actuators Topic List:
            MoveIt: Send the joint positions to the robot.
            /ned2/niryo_robot_follow_joint_trajectory_controller/command: arm trajectory controller.
            /ned2/gazebo_tool_commander/follow_joint_trajectory: gripper action server (sim-only; niryo_robot_tools_commander is bypassed in sim, see move_gripper_joints).
        """
        rospy.loginfo("Start Init NED2RobotGoalEnv Multiros!")

        if ros_port is not None:
            ros_common.change_ros_gazebo_master(ros_port=ros_port, gazebo_port=gazebo_port)

        self.gripper = gripper
        self.real_time = real_time  # if True, the simulation will run in real time

        # we don't need to pause/unpause gazebo if we are running in real time
        if self.real_time:
            unpause_pause_physics = False
        else:
            unpause_pause_physics = True

        if not self.real_time:
            gazebo_core.unpause_gazebo()

        spawn_robot = True

        # location of the robot URDF file
        # See ned2_robot_sim.py for the full rationale on these paths.
        urdf_pkg_name = "niryo_ned2_description_extras"
        urdf_file_name = "ned2_kinect.urdf.xacro" if not gripper else "ned2_kinect_gripper.urdf.xacro"
        urdf_folder = "/urdf"

        # extra urdf args
        urdf_xacro_args = None  # we don't have any in the env

        # namespace of the robot
        namespace = "/ned2"

        # robot state publisher
        robot_state_publisher_max_freq = None  # we don't change the publishing freq
        new_robot_state_term = False

        robot_model_name = "ned2"
        robot_ref_frame = "world"

        # Set the initial pose of the robot model
        robot_pos_x = 0.0
        robot_pos_y = 0.0
        robot_pos_z = 0.0 if not load_table else 0.78
        robot_ori_w = 1.0
        robot_ori_x = 0.0
        robot_ori_y = 0.0
        robot_ori_z = 0.0

        # Controllers config — see ned2_robot_sim.py for the full rationale.
        # Single source of truth lives in niryo_ned2_description_extras.
        controller_package_name = "niryo_ned2_description_extras"
        controllers_file = "ned2_controllers.yaml" if not gripper else "ned2_controllers_w_gripper.yaml"
        controllers_list = ["joint_state_controller", "niryo_robot_follow_joint_trajectory_controller"] if not gripper else \
            ["joint_state_controller", "niryo_robot_follow_joint_trajectory_controller", "gazebo_tool_commander"]

        self.load_table = load_table

        if load_table:
            gazebo_models.spawn_sdf_model_gazebo(pkg_name="reactorx200_description", file_name="model.sdf",
                                                 model_folder="/models/table",
                                                 model_name="table", namespace=namespace,
                                                 pos_x=0.2)

            # above function pauses the simulation, so we need to unpause it for real-time
            if self.real_time:
                gazebo_core.unpause_gazebo()

        # spawn a cube
        if load_cube:
            gazebo_models.spawn_sdf_model_gazebo(pkg_name="reactorx200_description", file_name="block.sdf",
                                                 model_folder="/models/block",
                                                 model_name="red_cube", namespace=namespace,
                                                 pos_x=0.35,
                                                 pos_z=0.795 if load_table else 0.015)

            # above function pauses the simulation, so we need to unpause it for real-time
            if self.real_time:
                gazebo_core.unpause_gazebo()

        reset_controllers = False

        reset_mode = "world"

        sim_step_mode = 1
        num_gazebo_steps = 1

        gazebo_max_update_rate = None
        gazebo_timestep = None

        if rospy.has_param('/ned2/gazebo_update_rate_multiplier'):
            gazebo_max_update_rate = rospy.get_param('/ned2/gazebo_update_rate_multiplier')
            rospy.loginfo(f"Applied Gazebo update_rate_multiplier = {gazebo_max_update_rate}")

        if rospy.has_param('/ned2/gazebo_time_step'):
            gazebo_timestep = rospy.get_param('/ned2/gazebo_time_step')
            rospy.loginfo(f"Applied Gazebo time_step = {gazebo_timestep}")

        kill_rosmaster = True

        kill_gazebo = True

        clean_logs = False

        # Pre-load controllers YAML (with PID gains) under /ned2 BEFORE
        # gazebo spawns the model. The gazebo_ros_control plugin reads
        # /ned2/gazebo_ros_control/pid_gains/joint_* at plugin-init time
        # (when the model is spawned). multiros's spawn_robot_in_gazebo
        # loads the controllers YAML AFTER spawn, which is too late —
        # the plugin has already given up and logged "No p gain
        # specified for pid". Without PIDs the arm sags under gravity
        # and MoveIt can't converge.
        ros_common.ros_load_yaml(
            pkg_name=controller_package_name,
            file_name=controllers_file,
            ns="/" + namespace.lstrip("/"),
        )

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
            unpause_pause_physics=unpause_pause_physics, action_cycle_time=action_cycle_time,
            controller_package_name=controller_package_name)

        # ---------- joint state
        if namespace is not None and namespace != '/':
            self.joint_state_topic = namespace + "/joint_states"
        else:
            self.joint_state_topic = "/joint_states"

        self.joint_state_sub = rospy.Subscriber(self.joint_state_topic, JointState, self.joint_state_callback)
        self.joint_state = JointState()

        # ---------- Moveit
        # Use the description-extras wrapper, which includes Niryo's
        # move_group.launch under <group ns="ned2"> so move_group ends
        # up at /ned2/move_group/... and MoveitMultiros(ns="ned2") can
        # find its action servers. Without the wrap, Niryo's launch
        # runs at root and the env hangs on the readiness check.
        ros_common.ros_launch_launcher(
            pkg_name="niryo_ned2_description_extras",
            launch_file_name="ned2_move_group.launch",
            args=[
                "hardware_version:=ned2",
                "simulation_mode:=true",
                "load_robot_description:=false",
                f"gripper:={'true' if gripper else 'false'}",
            ],
        )

        # ---------- kinect
        self.use_camera = use_camera

        # todo: check the camera topics
        if self.use_camera:
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

        # ---------- Niryo built-in wrist camera (opt-in, default off).
        # See ned2_robot_sim.py for the full rationale.
        self.use_wrist_camera = use_wrist_camera

        if self.use_wrist_camera:
            self.wrist_camera_rgb_sub = rospy.Subscriber("/gazebo_camera/image_raw", Image,
                                                          self.wrist_camera_rgb_callback)
            self.wrist_camera_rgb = Image()
            self.cv_image_wrist = None

        self._check_connection_and_readiness()

        self.arm_joint_names = ["joint_1",
                                "joint_2",
                                "joint_3",
                                "joint_4",
                                "joint_5",
                                "joint_6"]

        self.gripper_joint_names = ["joint_base_to_mors_1",
                                    "joint_base_to_mors_2"]

        if self.real_time:
            # we don't need to pause/unpause gazebo if we are running in real time
            self.move_NED2_object = MoveitMultiros(arm_name='arm',
                                                    robot_description="ned2/robot_description",
                                                    ns="ned2", pause_gazebo=False)
        else:
            self.move_NED2_object = MoveitMultiros(arm_name='arm',
                                                    robot_description="ned2/robot_description",
                                                    ns="ned2")

        # low-level control
        # rostopic for arm trajectory controller
        self.arm_controller_pub = rospy.Publisher('/ned2/niryo_robot_follow_joint_trajectory_controller/command',
                                                      JointTrajectory,
                                                      queue_size=10)

        # parameters for calculating FK, IK
        # tool_link, not wrist_link: 6-element action vector needs a
        # 6-joint chain. base_link → wrist_link has 5 joints; base_link
        # → tool_link has 6. tool_link also matches Niryo's SRDF
        # planning-group EE.
        self.ee_link = "tool_link"
        self.ref_frame = "base_link"

        # Fk with pykdl_utils - old method
        self.pykdl_robot = URDF.from_parameter_server(key='ned2/robot_description')
        self.kdl_kin = KDLKinematics(urdf=self.pykdl_robot, base_link=self.ref_frame, end_link=self.ee_link)

        # with ros_kinematics
        self.ros_kin = ros_kinematics.Kinematics_pyrobot(robot_description_parm="ned2/robot_description",
                                                         base_link=self.ref_frame,
                                                         end_link=self.ee_link)

        # Per-link FK chains for the safety check in _check_action_links_safe.
        # See NED2RobotEnv (sim) for the full rationale — short version:
        # PyKDL's ChainFkSolverPos_recursive expects len(q) to match the
        # *subchain* joint count, not the full arm DOF. Caching one
        # KDLKinematics per check-link lets us slice q[:n] correctly.
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

        if not self.real_time:
            gazebo_core.pause_gazebo()
        else:
            gazebo_core.unpause_gazebo()  # this is because loading models will pause the simulation
        rospy.loginfo("End Init NED2RobotEnv")

    # ---------------------------------------------------
    #   Custom methods for the NED2 Robot Environment

    # Link chain that _check_action_links_safe walks to predict whether a
    # candidate joint target would dip any arm link below the table. Order
    # matches the URDF chain shoulder→tool. joint5_motor (small fixed
    # mounting link) and hand_link (downstream of wrist via fixed joints)
    # are skipped — wrist_link + tool_link bracket them.
    SAFETY_CHECK_LINKS = (
        "shoulder_link",
        "arm_link",
        "elbow_link",
        "forearm_link",
        "wrist_link",
        "tool_link",
    )

    def _check_action_links_safe(self, joint_targets, current_joints=None):
        """
        Predict each arm link's world z under ``joint_targets`` and reject
        the action if any link would dip below ``table_z + safety_z_margin``.
        Also caps |target - current| per joint at ``max_joint_delta``.

        Rosparams (all under ``/ned2/``, with strict variants where the
        real value should be tighter):
          table_z, safety_z_margin[_strict], max_joint_delta[_strict]

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
            rospy.loginfo(f"[SAFETY] call #{self._safety_log_count}: floor={floor:.3f}, {zs}")

        return True, None

    def get_model_pose(self, model_name="red_cube", rpy=True):
        """
        Get the pose of an object in Gazebo.

        Args:
            model_name: name of the object whose pose is to be retrieved
            rpy: True if the orientation is to be returned as euler
                angles (default: True)

        Returns:
            success: True if the Gazebo lookup succeeded
            position: object position as a numpy float32 (x, y, z) array,
                or None on failure
            orientation: object orientation as a numpy float32 array
                (roll, pitch, yaw in radians when rpy=True, otherwise
                quaternion (x, y, z, w)), or None on failure
        """

        if not self.real_time:
            gazebo_core.unpause_gazebo()

        # relative_entity_name uses Gazebo's "<model>/<link>" convention
        # (URDF links are bare, but the link-state service wants the
        # model-qualified form).
        header, pose, twist, success = gazebo_models.gazebo_get_model_state(model_name=model_name,
                                                                            relative_entity_name="ned2/base_link")

        if not self.real_time:
            gazebo_core.pause_gazebo()

        if success:
            if rpy:
                orientation = euler_from_quaternion(
                    [pose.orientation.x, pose.orientation.y,
                     pose.orientation.z, pose.orientation.w])
                orientation = np.array(orientation, dtype=np.float32)
            else:
                orientation = np.array([pose.orientation.x, pose.orientation.y,
                                        pose.orientation.z, pose.orientation.w],
                                       dtype=np.float32)

            position = np.array([pose.position.x, pose.position.y, pose.position.z],
                                dtype=np.float32)

            return success, position, orientation

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
        done = gazebo_models.spawn_sdf_model_gazebo(pkg_name="reactorx200_description", file_name="block.sdf",
                                                    model_folder="/models/block",
                                                    model_name="red_cube", namespace="/ned2",
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

            # joint names — used below to pull positions/velocities by
            # name rather than trusting the driver's publish order.
            self.joint_state_names = list(joint_state.name)

            # Build the obs-facing joint vectors by NAME lookup so a
            # driver change that re-orders /joint_states (or adds an
            # extra finger / mimic joint) doesn't silently scramble the
            # observation. The expected joint set is arm + (gripper
            # when the gripper URDF is loaded).
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

    # Mors prismatic limits (URDF: ±0.01 m). See ned2_robot_sim.py for
    # the full rationale of the sim-direct-to-gazebo_tool_commander
    # gripper path.
    _MORS_OPEN_POS = 0.01
    _MORS_CLOSED_POS = -0.01
    _MORS_MOVE_SECS = 1.0

    def move_gripper_joints(self, action: str) -> bool:
        """
        Drive the gripper to "open" or "close" (binary, same API the
        real env exposes via niryo_robot_tools_commander).

        Sim path: publish a JointTrajectory to
        ``/gazebo_tool_commander/follow_joint_trajectory`` directly.
        See ned2_robot_sim.move_gripper_joints docstring for the full
        rationale.

        Args:
            action: "open" or "close".

        Returns:
            True if the action server returned a result.
        """
        target = self._MORS_OPEN_POS if action == "open" else self._MORS_CLOSED_POS

        # Controllers spawn under /ned2; the action server lives at
        # /ned2/gazebo_tool_commander/..., not at root.
        client = actionlib.SimpleActionClient(
            '/ned2/gazebo_tool_commander/follow_joint_trajectory',
            FollowJointTrajectoryAction,
        )
        if not client.wait_for_server(timeout=rospy.Duration(10.0)):
            rospy.logerr(
                "[NED2] gazebo_tool_commander action server not reachable at "
                "/ned2/gazebo_tool_commander/follow_joint_trajectory after 10 s."
            )
            return False

        goal = FollowJointTrajectoryGoal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = list(self.gripper_joint_names)
        point = JointTrajectoryPoint()
        point.positions = [target, target]
        point.velocities = [0.0, 0.0]
        point.accelerations = [0.0, 0.0]
        point.time_from_start = rospy.Duration(self._MORS_MOVE_SECS)
        goal.trajectory.points.append(point)

        client.send_goal(goal)
        if not client.wait_for_result(timeout=rospy.Duration(self._MORS_MOVE_SECS + 3.0)):
            rospy.logwarn(
                f"[NED2] gripper '{action}' trajectory did not return a "
                f"result within {self._MORS_MOVE_SECS + 3.0:.1f} s; continuing."
            )
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
            return self.move_NED2_object.set_trajectory_joints(q_positions, async_move=True)
        else:
            return self.move_NED2_object.set_trajectory_joints(q_positions)

    def set_trajectory_ee(self, pos: np.ndarray) -> bool:
        """
        Set a pose target for the end effector of the robot arm using moveit.
        """
        if self.real_time:
            # do not wait for the action to finish
            return self.move_NED2_object.set_trajectory_ee(position=pos, async_move=True)
        else:
            return self.move_NED2_object.set_trajectory_ee(position=pos)

    def get_ee_pose(self):
        """
        Returns the end-effector pose as a geometry_msgs/PoseStamped message

        This gives us the best pose if we are using the moveit config of the ReactorX repo
        They are getting the pose with ee_gripper_link
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
        Callback for Niryo's built-in wrist camera (sim, on `camera_link`).
        Topic: /gazebo_camera/image_raw — sensor_msgs/Image. Converts to
        an OpenCV RGB numpy array on self.cv_image_wrist.
        """
        self.wrist_camera_rgb = img_msg
        bridge = CvBridge()
        cv_image_bgr = bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        self.cv_image_wrist = cv2.cvtColor(cv_image_bgr, cv2.COLOR_BGR2RGB)

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
        rospy.wait_for_service("/ned2/move_group/trajectory_execution/set_parameters")
        rospy.logdebug(rostopic.get_topic_type("/ned2/planning_scene", blocking=True))
        rospy.logdebug(rostopic.get_topic_type("/ned2/move_group/status", blocking=True))

        return True

    # helper fn for _check_connection_and_readiness
    def _check_ros_controllers_ready(self):
        """
        Function to check if ros controllers are running
        """
        rospy.logdebug(rostopic.get_topic_type("/ned2/niryo_robot_follow_joint_trajectory_controller/state", blocking=True))

        return True

    def _check_camera_ready(self):
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

        if self.use_camera:
            self._check_camera_ready()

        rospy.loginfo("All system are ready!")

        return True
