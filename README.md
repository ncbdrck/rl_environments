# RL Environments Based on UniROS

[![Documentation Status](https://readthedocs.org/projects/uniros/badge/?version=latest)](https://uniros.readthedocs.io/en/latest/?badge=latest)

📚 **Full documentation**: [uniros.readthedocs.io](https://uniros.readthedocs.io/) (ecosystem-wide docs hosted via UniROS)

This repository contains sim and real prebuild environments (`Gymnasium-based`) created using [RealROS](https://github.com/ncbdrck/realros) and [MultiROS](https://github.com/ncbdrck/multiros) frameworks. 

Robots
- Trossen Robotics ReactorX-200 - [Documentation](https://docs.trossenrobotics.com/interbotix_xsarms_docs/specifications/rx200.html)
- Trossen Robotics ViperX-300S - [Documentation](https://docs.trossenrobotics.com/interbotix_xsarms_docs/specifications/vx300s.html)
- [Niryo Ned 2](https://niryo.com/product/educational-desktop-robotic-arm/) - [Documentation](https://docs.niryo.com/robots/ned2/) - [ROS Documentation](https://niryorobotics.github.io/ned_ros/)
- [Universal Robots UR5e](https://www.universal-robots.com/products/ur5-robot/) + Robotiq 2F-85 gripper - [ROS Documentation](http://wiki.ros.org/universal_robot)

# Prerequisites

Before installing this package, make sure you have the following prerequisites:

## Option 1: One-shot installer (recommended for fresh setups)

If you're starting fresh on Ubuntu 20.04, run the bootstrap installer
that ships in this repo (and identically in every other ecosystem
repo: UniROS, MultiROS, RealROS, sb3_ros_support,
rl_training_validation). It installs ROS Noetic, UniROS (with
MultiROS + RealROS as submodules), sb3_ros_support, this package
plus all 4 robots' vendor packages + supporting description-extras
+ cube tracker, and rl_training_validation.

```bash
git clone https://github.com/ncbdrck/rl_environments.git /tmp/uniros_bootstrap
bash /tmp/uniros_bootstrap/install_uniros_stack.sh                # interactive
bash /tmp/uniros_bootstrap/install_uniros_stack.sh -y             # unattended
bash /tmp/uniros_bootstrap/install_uniros_stack.sh -p ~/my_ws -y  # custom path
```

The script asks once whether to install all components or pick per-
component (ROS / UniROS / sb3_ros_support / rl_environments /
rl_training_validation). It refuses to run on anything other than
Ubuntu 20.04 because ROS Noetic doesn't officially support other
distros.

**Don't have Ubuntu 20.04?** A Docker image is shipped under
[`docker/`](docker/) — see [`docker/README.md`](docker/README.md)
for build / run / GPU / hardware-passthrough instructions. Works on
Ubuntu 22.04 / 24.04 hosts, on WSL2, and on machines with GPUs that
have no Ubuntu 20.04 driver.

## Option 2: Manual Installation of Prerequisites

### 1. UniROS

This ROS repo requires **UniROS**. It has the `RealROS` to train or evaluate the tasks in the real world. 
Then to simulate the tasks in Gazebo, it has `MultiROS` package. 


Please follow the instructions in the [UniROS repository](https://github.com/ncbdrck/UniROS) to install UniROS.

**Note:** Make sure to check out the `gymnasium` branch of the repositories before making the ros workspace.


### 2. Rx200 Robot Repository

You can download the official repository of the Rx200 robot from [here](https://github.com/Interbotix/interbotix_ros_manipulators) and follow the instructions to install it.

Furthermore, you can also follow the instructions in the [Rx200 Official Documentation](https://docs.trossenrobotics.com/interbotix_xsarms_docs/ros_interface/ros1/software_setup.html) to install the related ros packages.

At the moment, these are the installation instructions for the Rx200 robot with ROS Noetic on Ubuntu 20.04:

```shell
sudo apt install curl
curl 'https://raw.githubusercontent.com/Interbotix/interbotix_ros_manipulators/main/interbotix_ros_xsarms/install/amd64/xsarm_amd64_install.sh' > xsarm_amd64_install.sh
chmod +x xsarm_amd64_install.sh
./xsarm_amd64_install.sh -d noetic -p ~/catkin_ws
```
**Note**: This will also install ROS Noetic (if not already installed) and create a new ROS workspace in your home directory if `-p` is not used. So source your workspace accordingly.


### 3. Ned2 Robot Repository

Use the following instructions to install the Ned2 robot repository.

```bash
# Download the ROS workspace
cd ~/catkin_ws/src
git clone https://github.com/NiryoRobotics/ned_ros.git
cd ned_ros
git submodule update --init ros-foxglove-bridge
pip install -r requirements.txt
sudo apt install sqlite3 ffmpeg build-essential -y

# Install the dependencies
cd ~/catkin_ws/
rosdep update
rosdep install --from-paths src --ignore-src -r -y --skip-keys "python-rpi.gpio"

# Build the workspace
catkin_make
source devel/setup.bash
```

### 3a. Ned2 sim extras (`niryo_ned2_description_extras`)

For Ned2 **simulation** envs (reach / push / pnp sim) we also need a
small description-extras package that mounts the Ned2 on the same
desk model the RX200 sim uses + adds a head-mount Kinect v2 + brings
up `ros_control` so `niryo_robot_follow_joint_trajectory_controller`
works in Gazebo. Mirrors the role of `reactorx200_description` for
the RX200.

```bash
cd ~/catkin_ws/src
git clone https://github.com/ncbdrck/niryo_ned2_description_extras.git
cd ~/catkin_ws
catkin_make
source devel/setup.bash

# Verify the sim bring-up works standalone (no RL env):
roslaunch niryo_ned2_description_extras ned2_gazebo.launch                # reach / push
roslaunch niryo_ned2_description_extras ned2_gazebo.launch gripper:=true  # pnp
```

Not needed for **real** Ned2 envs — those bring up the Niryo driver
via `niryo_robot_bringup` instead.

### 3b. ViperX-300S sim extras (`viperx300s_description`)

For VX300S **simulation** reach envs we use a local description-extras
package that mirrors the RX200 tabletop setup: ViperX-300S on the cafe
table, optional red cube, Kinect v2 mount, and `ros_control` controllers
for Gazebo. Mirrors the role of `reactorx200_description` for the RX200.

```bash
cd ~/catkin_ws/src
git clone https://github.com/ncbdrck/viperx300s_description.git
cd ~/catkin_ws
catkin_make
source devel/setup.bash

# Verify the standalone scene with table + red cube (no RL env):
roslaunch viperx300s_description vx300s_gazebo.launch load_cube:=true
```

Not needed for **real** VX300S envs — the Interbotix driver handles
hardware bring-up.

### 3c. UR5e sim extras (`ur5e_description_extras`)

For UR5e **simulation** envs we ship a local description-extras
package that wraps the upstream UR5e + Robotiq 2F-85 URDF, mounts the
arm on a 4-legged `ur5_base` (~0.59 m) with a `cafe_table` next to it,
adds a head-mount Kinect v2, and brings up `ros_control` with PID
gains under `/ur5e/gazebo_ros_control/pid_gains`. Mirrors
`viperx300s_description` for UR5e geometry.

```bash
cd ~/catkin_ws/src
git clone https://github.com/ncbdrck/ur5e_description_extras.git
cd ~/catkin_ws
catkin_make
source devel/setup.bash

# Verify the standalone scene (ur5_base + cafe_table + arm + Kinect):
roslaunch ur5e_description_extras ur5e_gazebo.launch
```

For real UR5e envs, the description-extras package is expected to
ship a `ur5e_real.launch` wrapper that brings up `ur_robot_driver` +
a Robotiq driver + MoveIt under `/ur5e`. Wire that up at the lab
before running the `*_real-v0` envs.

Please note that the instructions assume you are using Ubuntu 20.04 and ROS Noetic. If you are using a different operating system or ROS version, make sure to adapt the commands accordingly.

# Installation

Follow these steps to install this package:

1. Clone the repository:
    ```shell
    cd ~/catkin_ws/src
    git clone https://github.com/ncbdrck/rl_environments.git
    ```

2. Clone supporting packages (per-robot description extras + the
   cube-pose tracker used by real push / pick-and-place envs):
    ```shell
    cd ~/catkin_ws/src

    # Shared sensor models (Kinect v2, ZED 2, etc.)
    git clone https://github.com/ncbdrck/common-sensors.git

    # Per-robot description extras (table, cube, vendor URDF wraps)
    git clone https://github.com/ncbdrck/reactorx200_description.git
    git clone https://github.com/ncbdrck/niryo_ned2_description_extras.git
    git clone https://github.com/ncbdrck/viperx300s_description.git
    git clone https://github.com/ncbdrck/ur5e_description_extras.git

    # Real-side cube-pose publisher (push / PnP real envs)
    git clone https://github.com/ncbdrck/rl_envs_cube_tracker.git
    ```

   If you used `install_uniros_stack.sh` to bootstrap the workspace,
   these are already cloned for you — this list is for users doing a
   fully manual install.

3. This package relies on several Python packages. You can install them by running the following command:

    ```shell
    # Install pip if you haven't already by running this command
    sudo apt-get install python3-pip

    # install the required Python packages by running
    cd ~/catkin_ws/src/rl_environments/
    pip3 install -r requirements.txt
    ```
4. Build the ROS packages and source the environment:
    ```shell
   cd ~/catkin_ws/
   rosdep install --from-paths src --ignore-src -r -y
   catkin_make
   source devel/setup.bash
   rospack profile
    ```

# Usage

Usage is similar to `openai_gym` or `gymnasium` environments. You can use the following code to create an environment and run it:

```python
#!/bin/python3
import sys

# ROS packages required
import rospy

# gym
import uniros as gym
# import gymnasium as gym  # alternative import if you only work with one environment at a time
import numpy as np

# We can use the following import statement if we want to use the realros package
import realros
from realros.utils import ros_common

# import the environment
import rl_environments

# wrappers
from realros.wrappers.normalize_action_wrapper import NormalizeActionWrapper
from realros.wrappers.normalize_obs_wrapper import NormalizeObservationWrapper
from realros.wrappers.time_limit_wrapper import TimeLimitWrapper

if __name__ == '__main__':
    # Kill all processes related to previous runs
    # ros_common.kill_all_ros_and_gazebo()

    # Clear ROS logs
    # ros_common.clean_ros_logs()

    # --- normal environments
    env = gym.make('RX200ReacherSim-v0', gazebo_gui=True, ee_action_type=False, seed=10,
                   delta_action=True, environment_loop_rate=10.0, action_cycle_time=0.500,
                   use_smoothing=False, action_speed=0.100, reward_type="dense", log_internal_state=False)

    # VX300S reach follows the same surface; pass load_cube=True when you
    # want the tabletop red cube in the sim scene.
    # env = gym.make('VX300SReacherSim-v0', gazebo_gui=True, ee_action_type=False, seed=10,
    #                delta_action=True, environment_loop_rate=10.0, action_cycle_time=0.500,
    #                use_smoothing=False, action_speed=0.100, reward_type="dense",
    #                log_internal_state=False, load_cube=True)

    # # --- goal environments
    # env = gym.make('RX200ReacherGoalSim-v0', gazebo_gui=False, ee_action_type=False, seed=10,
    #                delta_action=True, environment_loop_rate=10.0, action_cycle_time=0.500,
    #                use_smoothing=False, action_speed=0.100, reward_type="sparse", log_internal_state=False)

    # Normalize action space
    env = NormalizeActionWrapper(env)

    # Normalize observation space
    env = NormalizeObservationWrapper(env)
    # env = NormalizeObservationWrapper(env, normalize_goal_spaces=True)  # goal-conditioned environments

    # Set max steps
    env = TimeLimitWrapper(env, max_episode_steps=100)

    # reset the environment
    env.reset()

    # run the training loop
    for i in range(100):
        # get the action from the agent
        action = env.action_space.sample()

        # step the environment
        obs, reward, terminated, truncated, info = env.step(action)

        # print the observation and reward
        print(f"Step: {i}, Obs: {obs}, Reward: {reward}, terminated: {terminated}, truncated: {truncated}, Info: {info}")

        # if done, reset the environment
        if truncated or terminated:
            env.reset()

    # close the environment
    env.close()
    sys.exit()

```

For another example of using the environment with [ROS-Based Stable Baselines3](https://github.com/ncbdrck/sb3_ros_support), you can refer to the [rl_training_validation repository](https://github.com/ncbdrck/rl_training_validation).

The environments are standard Gymnasium environments. Any Gymnasium-compatible RL library (Stable Baselines3, CleanRL, Tianshou, RLlib, custom loop) works; `sb3_ros_support` is one convenience option.

## Documentation

Full ecosystem documentation: [uniros.readthedocs.io](https://uniros.readthedocs.io/).

## Contact

For questions, suggestions, or collaborations, feel free to reach out to the project maintainer at [j.kapukotuwa@research.ait.ie](mailto:j.kapukotuwa@research.ait.ie).
