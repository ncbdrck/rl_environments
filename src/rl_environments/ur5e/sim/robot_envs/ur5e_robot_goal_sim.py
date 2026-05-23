#!/bin/python3
"""
Goal-conditioned superclass for UR5e Robot environments (Gazebo).

Same shape as UR5eRobotEnv (ur5e_robot_sim) but inherits from
GazeboGoalEnv so subclasses can use Dict observation spaces with
``observation`` / ``achieved_goal`` / ``desired_goal`` keys (needed
by HER + the Gymnasium GoalEnv contract).

Side-effect imports ur5e_robot_sim to set GAZEBO_MODEL_PATH at module
import time (the env-internal Gazebo doesn't inherit env from roslaunch).
"""

from rl_environments.ur5e.sim.robot_envs import ur5e_robot_sim  # noqa: F401  (env side-effect)

from gymnasium.envs.registration import register
import numpy as np

from multiros.envs import GazeboGoalEnv

import rospy
import rostopic
from sensor_msgs.msg import JointState, Image
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from cv_bridge import CvBridge
import cv2

from multiros.utils import gazebo_core
from multiros.utils import gazebo_models
from multiros.utils.moveit_multiros import MoveitMultiros
from multiros.utils import ros_common
from multiros.utils import ros_kinematics

from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_kinematics import KDLKinematics
from tf.transformations import euler_from_quaternion


register(
    id='UR5eRobotGoalEnv-v0',
    entry_point='rl_environments.ur5e.sim.robot_envs.ur5e_robot_goal_sim:UR5eRobotGoalEnv',
    max_episode_steps=1000,
)


class UR5eRobotGoalEnv(GazeboGoalEnv.GazeboGoalEnv):
    """Goal-conditioned superclass. See ur5e_robot_sim.UR5eRobotEnv for the
    non-goal twin (same bring-up, same publish helpers)."""

    def __init__(self, ros_port: str = None, gazebo_port: str = None, gazebo_pid=None,
                 seed: int = None, real_time: bool = False, action_cycle_time=0.0,
                 load_cube: bool = False, use_kinect: bool = False):
        rospy.loginfo("Start Init UR5eRobotGoalEnv Multiros")

        if ros_port is not None:
            ros_common.change_ros_gazebo_master(ros_port=ros_port, gazebo_port=gazebo_port)

        self.real_time = real_time
        unpause_pause_physics = not self.real_time
        if not self.real_time:
            gazebo_core.unpause_gazebo()

        spawn_robot = True
        urdf_pkg_name = "ur5e_description_extras"
        urdf_file_name = "ur5e_robotiq85_kinect.urdf.xacro"
        urdf_folder = "/urdf"
        urdf_xacro_args = None
        namespace = "/ur5e"
        robot_state_publisher_max_freq = None
        new_robot_state_term = False
        robot_model_name = "ur5e"
        robot_ref_frame = "world"

        robot_pos_x = 0.0
        robot_pos_y = 0.0
        robot_pos_z = 0.59
        robot_ori_w = 1.0
        robot_ori_x = 0.0
        robot_ori_y = 0.0
        robot_ori_z = 0.0

        controller_package_name = "ur5e_description_extras"
        controllers_file = "ur5e_controller.yaml"
        controllers_list = ["joint_state_controller", "arm_controller", "gripper_controller"]
        ros_common.ros_load_yaml(
            pkg_name=controller_package_name,
            file_name=controllers_file,
            ns="/" + namespace.lstrip("/"),
        )

        if load_cube:
            gazebo_models.spawn_sdf_model_gazebo(
                pkg_name="ur5e_description_extras", file_name="block.sdf",
                model_folder="/models/block",
                model_name="red_cube", namespace=namespace,
                pos_x=0.40, pos_y=-0.20, pos_z=0.795,
            )
            if self.real_time:
                gazebo_core.unpause_gazebo()

        reset_controllers = False
        reset_mode = "world"
        sim_step_mode = 1
        num_gazebo_steps = 1

        gazebo_max_update_rate = None
        gazebo_timestep = None
        if rospy.has_param('/ur5e/gazebo_update_rate_multiplier'):
            gazebo_max_update_rate = rospy.get_param('/ur5e/gazebo_update_rate_multiplier')
        if rospy.has_param('/ur5e/gazebo_time_step'):
            gazebo_timestep = rospy.get_param('/ur5e/gazebo_time_step')

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
            gazebo_timestep=gazebo_timestep, kill_rosmaster=True, kill_gazebo=True,
            clean_logs=False, ros_port=ros_port, gazebo_port=gazebo_port, gazebo_pid=gazebo_pid, seed=seed,
            unpause_pause_physics=unpause_pause_physics, action_cycle_time=action_cycle_time,
            controller_package_name=controller_package_name)

        self.joint_state_topic = (namespace + "/joint_states"
                                  if namespace not in (None, '/') else "/joint_states")
        self.joint_state_sub = rospy.Subscriber(self.joint_state_topic, JointState,
                                                self.joint_state_callback)
        self.joint_state = JointState()

        ros_common.ros_launch_launcher(pkg_name="ur5e_description_extras",
                                       launch_file_name="ur5e_move_group.launch")

        self.use_kinect = use_kinect
        if self.use_kinect:
            self.kinect_depth_sub = rospy.Subscriber("/head_mount_kinect2/depth/image_raw",
                                                     Image, self.kinect_depth_callback)
            self.kinect_depth = Image()
            self.cv_image_depth = None
            self.kinect_rgb_sub = rospy.Subscriber("/head_mount_kinect2/rgb/image_raw",
                                                   Image, self.kinect_rgb_callback)
            self.kinect_rgb = Image()
            self.cv_image_rgb = None

        self._check_connection_and_readiness()

        self.arm_joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
        ]
        self.gripper_joint_names = ["robotiq_85_left_knuckle_joint"]

        self.move_UR5E_object = MoveitMultiros(
            arm_name='arm', gripper_name='gripper',
            robot_description="ur5e/robot_description", ns="ur5e",
            pause_gazebo=not self.real_time)

        self.arm_controller_pub = rospy.Publisher('/ur5e/arm_controller/command',
                                                  JointTrajectory, queue_size=10)
        self.gripper_controller_pub = rospy.Publisher('/ur5e/gripper_controller/command',
                                                      JointTrajectory, queue_size=10)

        self.ee_link = "ee_link"
        self.ref_frame = "base_link"

        self.pykdl_robot = URDF.from_parameter_server(key='ur5e/robot_description')
        self.kdl_kin = KDLKinematics(urdf=self.pykdl_robot,
                                     base_link=self.ref_frame, end_link=self.ee_link)
        self.ros_kin = ros_kinematics.Kinematics_pyrobot(
            robot_description_parm="ur5e/robot_description",
            base_link=self.ref_frame, end_link=self.ee_link)

        self._safety_kin = {}
        for _link in self.SAFETY_CHECK_LINKS:
            try:
                _kin = KDLKinematics(urdf=self.pykdl_robot,
                                     base_link=self.ref_frame, end_link=_link)
                self._safety_kin[_link] = (_kin, int(_kin.num_joints))
            except Exception as _e:
                rospy.logwarn(f"[SAFETY] kinematics setup failed for {_link}: {_e}")

        if not self.real_time:
            gazebo_core.pause_gazebo()
        else:
            gazebo_core.unpause_gazebo()
        rospy.loginfo("End Init UR5eRobotGoalEnv")

    # ---- helpers: same body as UR5eRobotEnv. Duplicated rather than
    # multi-inherited to keep the GazeboGoalEnv MRO simple.

    def get_model_pose(self, model_name="red_cube", rpy=True):
        if not self.real_time:
            gazebo_core.unpause_gazebo()
        header, pose, twist, success = gazebo_models.gazebo_get_model_state(
            model_name=model_name, relative_entity_name="ur5e/base_link")
        if not self.real_time:
            gazebo_core.pause_gazebo()
        if success:
            if rpy:
                orientation = np.array(euler_from_quaternion(
                    [pose.orientation.x, pose.orientation.y,
                     pose.orientation.z, pose.orientation.w]), dtype=np.float32)
            else:
                orientation = np.array([pose.orientation.x, pose.orientation.y,
                                        pose.orientation.z, pose.orientation.w],
                                       dtype=np.float32)
            position = np.array([pose.position.x, pose.position.y, pose.position.z],
                                dtype=np.float32)
            return success, position, orientation
        return success, None, None

    def spawn_cube_in_gazebo(self, model_pos_x, model_pos_y):
        done = gazebo_models.spawn_sdf_model_gazebo(
            pkg_name="ur5e_description_extras", file_name="block.sdf",
            model_folder="/models/block",
            model_name="red_cube", namespace="/ur5e",
            pos_x=model_pos_x, pos_y=model_pos_y, pos_z=0.795)
        if self.real_time:
            gazebo_core.unpause_gazebo()
        return done

    def remove_cube_in_gazebo(self):
        done = gazebo_models.remove_model_gazebo(model_name="red_cube")
        if self.real_time:
            gazebo_core.unpause_gazebo()
        return done

    def fk_pykdl(self, action):
        pose = self.kdl_kin.forward(action)
        return np.array([pose[0, 3], pose[1, 3], pose[2, 3]], dtype=np.float32)

    def calculate_fk(self, joint_positions, euler=True):
        return self.ros_kin.calculate_fk(joint_positions, des_frame=self.ee_link, euler=euler)

    SAFETY_CHECK_LINKS = (
        "shoulder_link", "upper_arm_link", "forearm_link",
        "wrist_1_link", "wrist_2_link", "wrist_3_link", "ee_link",
    )

    def _check_action_links_safe(self, joint_targets, current_joints=None):
        """See UR5eRobotEnv._check_action_links_safe for full rationale."""
        strict = bool(getattr(self, "enable_strict_safety", False))
        table_z = float(rospy.get_param("/ur5e/table_z", 0.59))
        if strict:
            margin = float(rospy.get_param("/ur5e/safety_z_margin_strict", 0.030))
            max_delta = float(rospy.get_param("/ur5e/max_joint_delta_strict", 0.15))
        else:
            margin = float(rospy.get_param("/ur5e/safety_z_margin", 0.015))
            max_delta = float(rospy.get_param("/ur5e/max_joint_delta", 0.5))
        floor = table_z + margin
        base_z = 0.59

        q = np.asarray(joint_targets, dtype=np.float64)
        if current_joints is not None:
            cur = np.asarray(current_joints, dtype=np.float64)
            if cur.shape == q.shape:
                deltas = np.abs(q - cur)
                if np.any(deltas > max_delta):
                    idx = int(np.argmax(deltas))
                    return False, f"joint[{idx}] delta {deltas[idx]:.3f} > {max_delta}"

        for link, (kin, n) in self._safety_kin.items():
            try:
                pose = kin.forward(q[:n])
            except Exception as e:
                return False, f"FK failed for {link}: {e}"
            z_world = float(pose[2, 3]) + base_z
            if z_world < floor:
                return False, f"{link} predicted z={z_world:.3f} < floor={floor:.3f}"

        return True, None

    def calculate_ik(self, target_pos, ee_ori=np.array([0.0, 0.0, 0.0, 1.0])):
        target_pose = np.concatenate((target_pos, ee_ori))
        ee_position = self.get_joint_angles()
        return self.ros_kin.calculate_ik(target_pose=target_pose, tolerance=[1e-3] * 6,
                                         init_joint_positions=ee_position)

    def joint_state_callback(self, joint_state):
        if joint_state is not None:
            self.joint_state = joint_state
            self.joint_state_names = list(joint_state.name)
            self.joint_pos_all = list(joint_state.position)
            self.current_joint_velocities = list(joint_state.velocity)
            self.current_joint_efforts = list(joint_state.effort)

    def move_arm_joints(self, q_positions: np.ndarray, time_from_start: float = 0.5) -> bool:
        traj = JointTrajectory()
        traj.joint_names = self.arm_joint_names
        traj.points.append(JointTrajectoryPoint())
        traj.points[0].positions = q_positions
        traj.points[0].velocities = [0.0] * len(self.arm_joint_names)
        traj.points[0].accelerations = [0.0] * len(self.arm_joint_names)
        traj.points[0].time_from_start = rospy.Duration(time_from_start)
        self.arm_controller_pub.publish(traj)
        return True

    def move_gripper_joints(self, q_positions: np.ndarray, time_from_start: float = 0.5) -> bool:
        traj = JointTrajectory()
        traj.joint_names = self.gripper_joint_names
        traj.points.append(JointTrajectoryPoint())
        traj.points[0].positions = q_positions
        traj.points[0].velocities = [0.0] * len(self.gripper_joint_names)
        traj.points[0].accelerations = [0.0] * len(self.gripper_joint_names)
        traj.points[0].time_from_start = rospy.Duration(time_from_start)
        self.gripper_controller_pub.publish(traj)
        return True

    def set_trajectory_joints(self, q_positions: np.ndarray) -> bool:
        if self.real_time:
            return self.move_UR5E_object.set_trajectory_joints(q_positions, async_move=True)
        return self.move_UR5E_object.set_trajectory_joints(q_positions)

    def set_trajectory_ee(self, pos: np.ndarray) -> bool:
        if self.real_time:
            return self.move_UR5E_object.set_trajectory_ee(position=pos, async_move=True)
        return self.move_UR5E_object.set_trajectory_ee(position=pos)

    def get_ee_pose(self):
        return self.move_UR5E_object.get_robot_pose()

    def get_ee_rpy(self):
        return self.move_UR5E_object.get_robot_rpy()

    def get_joint_angles(self):
        return self.move_UR5E_object.get_joint_angles_robot_arm()

    def check_goal(self, goal):
        return self.move_UR5E_object.check_goal(goal)

    def check_goal_reachable_joint_pos(self, joint_pos):
        return self.move_UR5E_object.check_goal_joint_pos(joint_pos)

    def kinect_depth_callback(self, data):
        self.kinect_depth = data
        bridge = CvBridge()
        self.cv_image_depth = bridge.imgmsg_to_cv2(data, desired_encoding="32FC1")

    def kinect_rgb_callback(self, img_msg):
        self.kinect_rgb = img_msg
        bridge = CvBridge()
        cv_image_bgr = bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        self.cv_image_rgb = cv2.cvtColor(cv_image_bgr, cv2.COLOR_BGR2RGB)

    def _check_joint_states_ready(self):
        if not self.real_time:
            gazebo_core.unpause_gazebo()
        rospy.logdebug(rostopic.get_topic_type(self.joint_state_topic, blocking=True))
        return True

    def _check_moveit_ready(self):
        rospy.wait_for_service("/ur5e/move_group/trajectory_execution/set_parameters")
        rospy.logdebug(rostopic.get_topic_type("/ur5e/planning_scene", blocking=True))
        rospy.logdebug(rostopic.get_topic_type("/ur5e/move_group/status", blocking=True))
        return True

    def _check_ros_controllers_ready(self):
        rospy.logdebug(rostopic.get_topic_type("/ur5e/arm_controller/state", blocking=True))
        rospy.logdebug(rostopic.get_topic_type("/ur5e/gripper_controller/state", blocking=True))
        return True

    def _check_kinect_ready(self):
        rospy.logdebug(rostopic.get_topic_type("/head_mount_kinect2/depth/points", blocking=True))
        return True

    def _check_connection_and_readiness(self):
        self._check_moveit_ready()
        self._check_joint_states_ready()
        self._check_ros_controllers_ready()
        if self.use_kinect:
            self._check_kinect_ready()
        rospy.loginfo("All system are ready!")
        return True
