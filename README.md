# rl_environments

## Installation of the NED environment
```bash
cd catkin_ws/src
git clone https://github.com/NiryoRobotics/ned_ros 
cd ..
rosdep update
rosdep install --from-paths src --ignore-src -r -y --skip-keys "python-rpi.gpio"
catkin_make
source devel/setup.bash
```

if you need to visualize the robot in rviz, you can use the default URDF visualization package of ROS
```bash
# install the package
sudo apt-get install ros-noetic-urdf-tutorial

# launch the visualization using above -urdf-tutorial package
roslaunch urdf_tutorial display.launch model:='$(find mycobot_description)/urdf/mycobot_280_m5/mycobot_280_m5.urdf'
```

## Installation of all the packages including ROS Noetic 

```bash
# Make the script executable
chmod +x install_ros_rl.sh

# Run the script interactively
./install_ros_rl.sh

# Or, run the script in non-interactive mode
./install_ros_rl.sh -n
```
