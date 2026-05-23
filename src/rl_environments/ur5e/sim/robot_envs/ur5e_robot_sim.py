#!/bin/python3
"""
Superclass for all UR5e Robot environments (Gazebo).

Mirrors vx300s_robot_sim / ned2_robot_sim / rx200_robot_sim in shape:
spawns the URDF under /ur5e, brings up ros_control, launches MoveIt
under the same namespace, exposes FK / IK / joint + gripper publish
helpers, and a per-link FK safety check.
"""

import os

import rospkg

from gymnasium.envs.registration import register
import numpy as np

from multiros.envs import GazeboBaseEnv

import rospy
import rostopic
from gazebo_msgs.srv import SetModelConfiguration
from sensor_msgs.msg import JointState, Image
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from cv_bridge import CvBridge
import cv2

# core modules of the framework
from multiros.utils import gazebo_core
from multiros.utils import gazebo_models
from multiros.utils.moveit_multiros import MoveitMultiros
from multiros.utils import ros_common
from multiros.utils import ros_controllers
from multiros.utils import ros_kinematics

from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_kinematics import KDLKinematics
from tf.transformations import euler_from_quaternion


# When the env launches its own Gazebo subprocess (multiros.GazeboBaseEnv
# path) it doesn't inherit GAZEBO_MODEL_PATH from roslaunch <env> tags —
# so we set it at module import time so model:// URIs in ur5e_scene.world
# (ur5_base, cafe_table) resolve.
_rp = rospkg.RosPack()


def _prepend_env_path(var: str, path: str) -> None:
    if not path:
        return
    cur = os.environ.get(var, "")
    if path in cur.split(":"):
        return
    os.environ[var] = f"{path}:{cur}" if cur else path


try:
    _prepend_env_path("GAZEBO_MODEL_PATH",
                      _rp.get_path("ur5e_description_extras") + "/models")
except rospkg.common.ResourceNotFound:
    pass


register(
    id='UR5eRobotBaseEnv-v0',
    entry_point='rl_environments.ur5e.sim.robot_envs.ur5e_robot_sim:UR5eRobotEnv',
    max_episode_steps=1000,
)


class UR5eRobotEnv(GazeboBaseEnv.GazeboBaseEnv):
    """Superclass for all UR5e Robot environments."""

    def __init__(self, ros_port: str = None, gazebo_port: str = None, gazebo_pid=None,
                 seed: int = None, real_time: bool = False, action_cycle_time=0.0,
                 load_cube: bool = False, load_table: bool = True, use_kinect: bool = False):
        """
        Sensor topics:
          MoveIt: pose + rpy of the robot.
          /ur5e/joint_states: arm + gripper joint states.
          /head_mount_kinect2/depth/image_raw: depth from head-mount Kinect v2.
          /head_mount_kinect2/rgb/image_raw:   rgb from head-mount Kinect v2.

        Actuator topics:
          MoveIt: set joint targets or EE pose via planning.
          /ur5e/arm_controller/command:     6-DOF arm trajectory commands.
          /ur5e/gripper_controller/command: Robotiq 2F-85 knuckle command.
        """
        rospy.loginfo("Start Init UR5eRobotEnv Multiros")

        if ros_port is not None:
            ros_common.change_ros_gazebo_master(ros_port=ros_port, gazebo_port=gazebo_port)

        self.real_time = real_time
        self.load_table = load_table

        if self.real_time:
            unpause_pause_physics = False
        else:
            unpause_pause_physics = True

        spawn_robot = True

        # URDF location. Our wrap declares its own world link + gazebo_ros_control
        # plugin under /ur5e (bypasses ur_e_description/common.gazebo.xacro
        # which hardcodes <robotNamespace>/</robotNamespace>).
        urdf_pkg_name = "ur5e_description_extras"
        urdf_file_name = "ur5e_robotiq85_kinect.urdf.xacro"
        urdf_folder = "/urdf"
        urdf_xacro_args = None

        namespace = "/ur5e"

        robot_state_publisher_max_freq = None
        new_robot_state_term = False

        robot_model_name = "ur5e"
        robot_ref_frame = "world"

        # Spawn pose. UR5e sits on top of the 4-legged ur5_base which is
        # baked into ur5e_scene.world; the base top plate is at z = 0.59.
        robot_pos_x = 0.0
        robot_pos_y = 0.0
        self.base_z = float(rospy.get_param("/ur5e/base_z", 0.59))
        robot_pos_z = self.base_z
        robot_ori_w = 1.0
        robot_ori_x = 0.0
        robot_ori_y = 0.0
        robot_ori_z = 0.0

        # UR5e all-zero joints put the arm horizontal through the cafe
        # table. Keep the env-internal spawn path aligned with
        # ur5e_gazebo.launch by forcing a folded, above-table pose before
        # physics is allowed to run.
        self.safe_init_pos = np.array([0.0, -1.5707, 1.5707, -1.5707, -1.5707, 0.0],
                                      dtype=np.float64)

        # Pre-load the controller YAML (with gazebo_ros_control pid_gains)
        # under /ur5e BEFORE the model spawns. gazebo_ros_control reads
        # /ur5e/gazebo_ros_control/pid_gains/joint_* at plugin-init time;
        # multiros's spawn_robot_in_gazebo loads the YAML AFTER spawn,
        # which is too late and gives "No p gain specified for pid".
        controller_package_name = "ur5e_description_extras"
        controllers_file = "ur5e_controller.yaml"
        controllers_list = ["joint_state_controller", "arm_controller", "gripper_controller"]
        ros_common.ros_load_yaml(
            pkg_name=controller_package_name,
            file_name=controllers_file,
            ns="/" + namespace.lstrip("/"),
        )

        # Optional cube. ur5e_scene.world doesn't bake the cube in (RL env
        # resets need to be able to re-spawn it).
        if load_cube:
            gazebo_models.spawn_sdf_model_gazebo(
                pkg_name="ur5e_description_extras", file_name="block.sdf",
                model_folder="/models/block",
                model_name="red_cube", namespace=namespace,
                pos_x=0.40, pos_y=-0.20, pos_z=0.795,
            )

        reset_controllers = False
        reset_mode = "world"
        sim_step_mode = 1
        num_gazebo_steps = 1

        gazebo_max_update_rate = None
        gazebo_timestep = None
        if rospy.has_param('/ur5e/gazebo_update_rate_multiplier'):
            gazebo_max_update_rate = rospy.get_param('/ur5e/gazebo_update_rate_multiplier')
            rospy.loginfo(f"Applied Gazebo update_rate_multiplier = {gazebo_max_update_rate}")
        if rospy.has_param('/ur5e/gazebo_time_step'):
            gazebo_timestep = rospy.get_param('/ur5e/gazebo_time_step')
            rospy.loginfo(f"Applied Gazebo time_step = {gazebo_timestep}")

        kill_rosmaster = True
        kill_gazebo = True
        clean_logs = False

        super().__init__(
            spawn_robot=spawn_robot, urdf_pkg_name=urdf_pkg_name, urdf_file_name=urdf_file_name,
            urdf_folder=urdf_folder, urdf_xacro_args=urdf_xacro_args, namespace=namespace,
            robot_state_publisher_max_freq=robot_state_publisher_max_freq, new_robot_state_term=new_robot_state_term,
            robot_model_name=robot_model_name, robot_ref_frame=robot_ref_frame,
            robot_pos_x=robot_pos_x, robot_pos_y=robot_pos_y, robot_pos_z=robot_pos_z, robot_ori_w=robot_ori_w,
            robot_ori_x=robot_ori_x, robot_ori_y=robot_ori_y, robot_ori_z=robot_ori_z,
            controllers_file=None, controllers_list=controllers_list,
            reset_controllers=reset_controllers, reset_mode=reset_mode, sim_step_mode=sim_step_mode,
            num_gazebo_steps=num_gazebo_steps, gazebo_max_update_rate=gazebo_max_update_rate,
            gazebo_timestep=gazebo_timestep, kill_rosmaster=kill_rosmaster, kill_gazebo=kill_gazebo,
            clean_logs=clean_logs, ros_port=ros_port, gazebo_port=gazebo_port, gazebo_pid=gazebo_pid, seed=seed,
            unpause_pause_physics=unpause_pause_physics,
            action_cycle_time=action_cycle_time if self.real_time else 0.0,
            controller_package_name=controller_package_name)

        # ---------- joint state
        if namespace is not None and namespace != '/':
            self.joint_state_topic = namespace + "/joint_states"
        else:
            self.joint_state_topic = "/joint_states"

        self.joint_state_sub = rospy.Subscriber(self.joint_state_topic, JointState,
                                                self.joint_state_callback)
        self.joint_state = JointState()
        self.joint_state_names = []
        self.joint_pos_all = []
        self.current_joint_velocities = []
        self.current_joint_efforts = []

        # ---------- Kinect (opt-in)
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

        self.arm_joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]

        # Robotiq 2F-85 exposes only the left knuckle joint as a real
        # actuator; the other finger joints follow through URDF mimic
        # linkage. So our gripper publish list has one joint.
        self.gripper_joint_names = ["robotiq_85_left_knuckle_joint"]

        # low-level control
        self.arm_controller_pub = rospy.Publisher('/ur5e/arm_controller/command',
                                                  JointTrajectory, queue_size=10)
        self.gripper_controller_pub = rospy.Publisher('/ur5e/gripper_controller/command',
                                                      JointTrajectory, queue_size=10)

        self._set_initial_model_configuration()
        gazebo_core.unpause_gazebo()
        self._spawn_ros_controllers(controllers_list, namespace)
        self.move_arm_joints(self.safe_init_pos.astype(np.float32), time_from_start=0.1)

        # ---------- MoveIt
        # Wrapper that runs move_group inside <group ns="ur5e"> so the
        # upstream ur5e_robotiq_85_moveit_config (un-namespaced) lands
        # at /ur5e/move_group/... matching robot_description there.
        ros_common.ros_launch_launcher(pkg_name="ur5e_description_extras",
                                       launch_file_name="ur5e_move_group.launch")

        self._check_connection_and_readiness()

        if self.real_time:
            self.move_UR5E_object = MoveitMultiros(arm_name='arm',
                                                   gripper_name='gripper',
                                                   robot_description="ur5e/robot_description",
                                                   ns="ur5e", pause_gazebo=False)
        else:
            self.move_UR5E_object = MoveitMultiros(arm_name='arm',
                                                   gripper_name='gripper',
                                                   robot_description="ur5e/robot_description",
                                                   ns="ur5e")

        # FK / IK. The MoveIt SRDF declares the arm chain as
        # base_link -> ee_link, so we use those bare URDF link names
        # here (no ur5e/ prefix — UR5e URDF links are bare unlike
        # Interbotix's vx300s/ etc).
        self.ee_link = "ee_link"
        self.ref_frame = "base_link"

        # pykdl_utils (legacy path used by FK helper)
        self.pykdl_robot = URDF.from_parameter_server(key='ur5e/robot_description')
        self.kdl_kin = KDLKinematics(urdf=self.pykdl_robot,
                                     base_link=self.ref_frame, end_link=self.ee_link)

        # ros_kinematics (preferred for IK)
        self.ros_kin = ros_kinematics.Kinematics_pyrobot(
            robot_description_parm="ur5e/robot_description",
            base_link=self.ref_frame, end_link=self.ee_link)

        # Per-link FK chains for _check_action_links_safe. See vx300s
        # robot env for full rationale: PyKDL needs len(q) == subchain
        # joint count, so we cache one KDLKinematics per check-link.
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
        rospy.loginfo("End Init UR5eRobotEnv")

    # ---------------------------------------------------
    # Custom methods

    def _set_initial_model_configuration(self) -> bool:
        """Set Gazebo's UR5e joints to the folded safe pose immediately after spawn."""
        try:
            rospy.wait_for_service("/gazebo/set_model_configuration", timeout=10.0)
            set_model_config = rospy.ServiceProxy("/gazebo/set_model_configuration",
                                                  SetModelConfiguration)
            resp = set_model_config(
                model_name="ur5e",
                urdf_param_name="/ur5e/robot_description",
                joint_names=self.arm_joint_names,
                joint_positions=[float(v) for v in self.safe_init_pos],
            )
        except Exception as exc:
            rospy.logwarn(f"Failed to set UR5e safe initial joint pose: {exc}")
            return False

        if not resp.success:
            rospy.logwarn(f"UR5e safe initial joint pose rejected: {resp.status_message}")
            return False

        rospy.loginfo("UR5e safe initial joint pose applied.")
        return True

    def _spawn_ros_controllers(self, controllers_list, namespace: str) -> bool:
        """Start controllers after the safe pose is applied and Gazebo is unpaused."""
        if ros_controllers.spawn_controllers(controllers_list, ns=namespace):
            rospy.loginfo("UR5e controllers spawned successfully.")
            return True

        rospy.logerr("Failed to spawn UR5e controllers.")
        return False

    def get_model_pose(self, model_name="red_cube", rpy=True):
        """Get an object's pose from Gazebo in the ur5e base frame."""
        if not self.real_time:
            gazebo_core.unpause_gazebo()

        # ur5e/base_link uses Gazebo's "<model>/<link>" convention;
        # the URDF link is bare "base_link" but Gazebo's link-state
        # lookup wants the model-qualified form.
        header, pose, twist, success = gazebo_models.gazebo_get_model_state(
            model_name=model_name, relative_entity_name="ur5e/base_link")

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
        """Spawn the red cube on the cafe table at z = 0.795."""
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
        """Forward kinematics via pykdl_utils. Returns EE position (np.ndarray)."""
        pose = self.kdl_kin.forward(action)
        return np.array([pose[0, 3], pose[1, 3], pose[2, 3]], dtype=np.float32)

    def calculate_fk(self, joint_positions, euler=True):
        """Forward kinematics via ros_kinematics. Returns (done, ee_pos, ee_rpy_or_quat)."""
        return self.ros_kin.calculate_fk(joint_positions, des_frame=self.ee_link, euler=euler)

    # Arm links whose world z must stay above the safety floor for the
    # action to be safe. base_link omitted (it's fixed; the world joint
    # is rigid). robotiq_* downstream of ee_link via fixed joints —
    # checking ee_link covers them.
    SAFETY_CHECK_LINKS = (
        "shoulder_link",
        "upper_arm_link",
        "forearm_link",
        "wrist_1_link",
        "wrist_2_link",
        "wrist_3_link",
        "ee_link",
    )

    def _check_action_links_safe(self, joint_targets, current_joints=None):
        """
        Predict each arm link's world z under ``joint_targets`` and reject
        the action if any link would dip below ``table_z + safety_z_margin``.
        Also caps |target - current| per joint at ``max_joint_delta``.

        UR5e mounts at z=0.59 on top of the ur5_base; the FK chain returns
        z values in the robot's base_link frame. The +0.59 base offset is
        added before comparing to the world-frame floor (table_z).

        Rosparams (all under ``/ur5e/``):
          table_z, safety_z_margin[_strict], max_joint_delta[_strict]

        Returns
        -------
        (safe, reason) : (bool, Optional[str])
        """
        strict = bool(getattr(self, "enable_strict_safety", False))
        base_z = float(rospy.get_param("/ur5e/base_z", self.base_z))
        floor_z = float(rospy.get_param("/ur5e/base_safety_z",
                                        rospy.get_param("/ur5e/table_z", base_z)))
        table_top_z = float(rospy.get_param("/ur5e/table_top_z", 0.775))
        table_center_x = float(rospy.get_param("/ur5e/table_center_x", 0.7))
        table_center_y = float(rospy.get_param("/ur5e/table_center_y", 0.0))
        table_size_x = float(rospy.get_param("/ur5e/table_size_x", 0.913))
        table_size_y = float(rospy.get_param("/ur5e/table_size_y", 0.913))
        table_xy_margin = float(rospy.get_param("/ur5e/table_xy_margin", 0.03))
        if strict:
            margin = float(rospy.get_param("/ur5e/safety_z_margin_strict", 0.030))
            max_delta = float(rospy.get_param("/ur5e/max_joint_delta_strict", 0.15))
        else:
            margin = float(rospy.get_param("/ur5e/safety_z_margin", 0.015))
            max_delta = float(rospy.get_param("/ur5e/max_joint_delta", 0.5))
        base_floor = floor_z + margin
        table_floor = table_top_z + margin
        table_x_min = table_center_x - (table_size_x / 2.0) - table_xy_margin
        table_x_max = table_center_x + (table_size_x / 2.0) + table_xy_margin
        table_y_min = table_center_y - (table_size_y / 2.0) - table_xy_margin
        table_y_max = table_center_y + (table_size_y / 2.0) + table_xy_margin

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
            x_world = float(pose[0, 3])
            y_world = float(pose[1, 3])
            z_world = float(pose[2, 3]) + base_z
            per_link_z.append((link, z_world))
            if z_world < base_floor:
                return False, f"{link} predicted z={z_world:.3f} < base_floor={base_floor:.3f}"
            over_table_xy = table_x_min <= x_world <= table_x_max and table_y_min <= y_world <= table_y_max
            if over_table_xy and z_world < table_floor:
                return False, f"{link} predicted z={z_world:.3f} < table_floor={table_floor:.3f}"

        if not hasattr(self, "_safety_log_count"):
            self._safety_log_count = 0
        if self._safety_log_count < 3:
            self._safety_log_count += 1
            zs = ", ".join(f"{l}={z:.3f}" for l, z in per_link_z)
            rospy.loginfo(
                f"[SAFETY] call #{self._safety_log_count}: base_floor={base_floor:.3f}, "
                f"table_floor={table_floor:.3f}, {zs}"
            )

        return True, None

    def calculate_ik(self, target_pos, ee_ori=np.array([0.0, 0.0, 0.0, 1.0])):
        """Inverse kinematics via ros_kinematics. Returns (done, joint_positions)."""
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
        """Send a single JointTrajectory point to the arm controller."""
        trajectory = JointTrajectory()
        trajectory.joint_names = self.arm_joint_names
        trajectory.points.append(JointTrajectoryPoint())
        trajectory.points[0].positions = q_positions
        trajectory.points[0].velocities = [0.0] * len(self.arm_joint_names)
        trajectory.points[0].accelerations = [0.0] * len(self.arm_joint_names)
        trajectory.points[0].time_from_start = rospy.Duration(time_from_start)
        self.arm_controller_pub.publish(trajectory)
        return True

    def move_gripper_joints(self, q_positions: np.ndarray, time_from_start: float = 0.5) -> bool:
        """Send a single JointTrajectory point to the Robotiq knuckle.

        Robotiq 2F-85 range: ~0.0 (fully open) to ~0.8 rad (fully closed).
        """
        trajectory = JointTrajectory()
        trajectory.joint_names = self.gripper_joint_names
        trajectory.points.append(JointTrajectoryPoint())
        trajectory.points[0].positions = q_positions
        trajectory.points[0].velocities = [0.0] * len(self.gripper_joint_names)
        trajectory.points[0].accelerations = [0.0] * len(self.gripper_joint_names)
        trajectory.points[0].time_from_start = rospy.Duration(time_from_start)
        self.gripper_controller_pub.publish(trajectory)
        return True

    def smooth_trajectory(self, q_positions, time_from_start, multiplier=100):
        """Interpolate from current to target joint pos with `multiplier` steps."""
        num_steps = int(time_from_start * multiplier)
        current_positions = self.joint_values
        delta_positions = (q_positions - current_positions) / num_steps
        trajectory_points = []
        for step in range(1, num_steps + 1):
            intermediate_positions = current_positions + step * delta_positions
            trajectory_points.append((intermediate_positions, time_from_start / num_steps * step))
        self.publish_trajectory(trajectory_points)
        return True

    def publish_trajectory(self, trajectory_points):
        trajectory = JointTrajectory()
        trajectory.joint_names = self.arm_joint_names
        for positions, time_from_start in trajectory_points:
            point = JointTrajectoryPoint()
            point.positions = positions
            point.velocities = [0.0] * len(self.arm_joint_names)
            point.accelerations = [0.0] * len(self.arm_joint_names)
            point.time_from_start = rospy.Duration(time_from_start)
            trajectory.points.append(point)
        self.arm_controller_pub.publish(trajectory)

    def set_trajectory_joints(self, q_positions: np.ndarray) -> bool:
        """MoveIt joint-space plan + execute."""
        if self.real_time:
            return self.move_UR5E_object.set_trajectory_joints(q_positions, async_move=True)
        return self.move_UR5E_object.set_trajectory_joints(q_positions)

    def set_trajectory_ee(self, pos: np.ndarray) -> bool:
        """MoveIt EE-pose plan + execute."""
        if self.real_time:
            return self.move_UR5E_object.set_trajectory_ee(position=pos, async_move=True)
        return self.move_UR5E_object.set_trajectory_ee(position=pos)

    def get_ee_pose(self):
        return self.move_UR5E_object.get_robot_pose()

    def get_ee_rpy(self):
        return self.move_UR5E_object.get_robot_rpy()

    def get_joint_angles(self):
        """Current arm joint positions (6 elements)."""
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

    # ---------- readiness helpers
    def _check_joint_states_ready(self):
        gazebo_core.unpause_gazebo()
        msg = rospy.wait_for_message(self.joint_state_topic, JointState, timeout=10.0)
        self.joint_state_callback(msg)
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
