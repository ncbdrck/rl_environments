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
from niryo_robot_tools_commander.msg import ToolAction, ToolGoal, ToolCommand
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
from multiros.utils import ros_kinematics

from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_kinematics import KDLKinematics
from tf.transformations import euler_from_matrix

"""
Although it is best to register only the task environment, one can also register the robot environment. 
This is not necessary, but we can see if this section 
(Load the robot to gazebo and can control the robot with moveit or ros controllers)
works by calling "gymnasium.make" this env.
but you need to
    1. run gazebo - gazebo_core.launch_gazebo(launch_roscore=False, paused=False, pub_clock_frequency=100, gui=True)
    2. init a node - rospy.init_node('test_MyRobotGoalEnv')
    3. gymnasium.make("NED2RobotEnv-v0")
"""
register(
    id='NED2RobotEnv-v0',
    entry_point='rl_environments.ned2.sim.robot_envs.ned2_robot_sim:NED2RobotEnv',
    max_episode_steps=1000,
)


class NED2RobotEnv(GazeboBaseEnv.GazeboBaseEnv):
    """
    Superclass for all NED2 Robot environments - gazebo-based.
    """

    def __init__(self, ros_port: str = None, gazebo_port: str = None, gazebo_pid=None, seed: int = None,
                 real_time: bool = False, action_cycle_time=0.0, load_cube: bool = False, load_table: bool = False,
                 use_camera: bool = False, gripper: bool = False):
        """
        Initializes a new Robot Environment

        Describe the robot and the sensors used in the env.

        Sensor Topic List:
            MoveIt: To get the pose and rpy of the robot.
            /joint_states: JointState received for the joints of the robot
            /gazebo_camera/image_raw: RGB image from the robot camera.

        Actuators Topic List:
            MoveIt: Send the joint positions to the robot.
            /ned2/niryo_robot_follow_joint_trajectory_controller/command: Send the joint positions to the robot.
            /ned2/niryo_robot_tools_commander/action_server: Action server to control the robot tools.
        """
        rospy.loginfo("Start Init NED2RobotEnv Multiros")

        """
        Change the ros/gazebo master
        """
        if ros_port is not None:
            ros_common.change_ros_gazebo_master(ros_port=ros_port, gazebo_port=gazebo_port)

        """
        parameters
        """
        self.gripper = gripper
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
        urdf_pkg_name = "niryo_robot_description"
        urdf_file_name = "niryo_ned2_gazebo.urdf.xacro" if not gripper else "niryo_ned2_gripper1_n_camera.urdf"
        urdf_folder = "/urdf/ned2"

        # extra urdf args
        urdf_xacro_args = None

        # namespace of the robot
        namespace = "ned2"

        # robot state publisher
        robot_state_publisher_max_freq = None
        new_robot_state_term = False

        robot_model_name = "ned2"
        robot_ref_frame = "world"

        # Set the initial pose of the robot model
        robot_pos_x = 0.0
        robot_pos_y = 0.0
        robot_pos_z = 0.0 if not load_table else 0.78
        robot_ori_w = 0.0
        robot_ori_x = 0.0
        robot_ori_y = 0.0
        robot_ori_z = 0.0

        # controller (must be inside above pkg_name/config/)
        controller_package_name = "rl_environments"
        controllers_file = "ned2_ros_controllers.yaml"
        controllers_list = ["joint_state_controller", "niryo_robot_follow_joint_trajectory_controller"] if not gripper else \
            ["joint_state_controller", "niryo_robot_follow_joint_trajectory_controller", "gazebo_tool_commander"]

        """
        Spawn other objects in Gazebo
        """
        # spawn a table
        self.load_table = load_table

        if load_table:
            # we can use the model from reactorx200_description package
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

        if rospy.has_param('/ned2/update_rate_multiplier'):
            gazebo_max_update_rate = rospy.get_param('/ned2/gazebo_update_rate_multiplier')

        if rospy.has_param('/ned2/time_step'):
            gazebo_timestep = rospy.get_param('/ned2/time_step')

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
            unpause_pause_physics=unpause_pause_physics, action_cycle_time=action_cycle_time,
            controller_package_name=controller_package_name)

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
        if not gripper:
            ros_common.ros_launch_launcher(pkg_name="niryo_moveit_config_standalone",
                                           launch_file_name="move_group.launch",
                                           args=["hardware_version:=ned2", "simulation_mode:=true",
                                                 "load_robot_description:=false"
                                                 ])
        else:
            ros_common.ros_launch_launcher(pkg_name="niryo_moveit_config_w_gripper1",
                                           launch_file_name="move_group.launch",
                                           args=["hardware_version:=ned2", "simulation_mode:=true",
                                                 "load_robot_description:=false"
                                                 ])

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

        """
        Using the _check_connection_and_readiness method to check for the connection status of subscribers, publishers 
        and services
        """

        self._check_connection_and_readiness()

        """
        initialise controller and sensor objects here
        """
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
        self.ee_link = "ned2/wrist_link"
        self.ref_frame = "ned2/base_link"

        # Fk with pykdl_utils - old method
        self.pykdl_robot = URDF.from_parameter_server(key='ned2/robot_description')
        self.kdl_kin = KDLKinematics(urdf=self.pykdl_robot, base_link=self.ref_frame, end_link=self.ee_link)

        # with ros_kinematics
        self.ros_kin = ros_kinematics.Kinematics_pyrobot(robot_description_parm="ned2/robot_description",
                                                         base_link=self.ref_frame,
                                                         end_link=self.ee_link)

        """
        Finished __init__ method
        """
        if not self.real_time:
            gazebo_core.pause_gazebo()
        else:
            gazebo_core.unpause_gazebo()  # this is because loading models will pause the simulation
        rospy.loginfo("End Init NED2RobotEnv")

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
        * get_joint_angles: Get current joint angles of the robot arm - 5 elements
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
                                                                            relative_entity_name="ned2/base_link")

        if not self.real_time:
            gazebo_core.pause_gazebo()

        if success:
            if rpy:
                orientation = euler_from_matrix([[pose.orientation.x, pose.orientation.y, pose.orientation.z],
                                                 [pose.orientation.y, pose.orientation.w, pose.orientation.z],
                                                 [pose.orientation.z, pose.orientation.y, pose.orientation.w]])

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
        done = gazebo_models.spawn_sdf_model_gazebo(pkg_name="reactorx200_description", file_name="block.sdf",
                                                    model_folder="/models/block",
                                                    model_name="red_cube", namespace="/rx200",
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

    def move_gripper_joints(self, action: str) -> bool:
        """
        Set a joint position target only for the gripper joints using low-level ros controllers. - ros action server

        Args:
            action: "open" or "close" the gripper

        Returns:
            True if the action is successful
        """

        # for the gripper
        gripper_action_client = actionlib.SimpleActionClient('/ned2/niryo_robot_tools_commander/action_server',
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
        get current joint angles of the robot arm - 5 elements
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
