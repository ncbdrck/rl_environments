#!/usr/bin/env bash

OFF='\033[0m'
RED='\033[0;31m'
GRN='\033[0;32m'
BLU='\033[0;34m'
BOLD=$(tput bold)
NORM=$(tput sgr0)
ERR="${RED}${BOLD}"
RRE="${NORM}${OFF}"
PROMPT="> "

function help() {
  cat << EOF
USAGE: ./install_rl_env.sh [-h][-p PATH][-n]

Install the RL environment and dependencies for ROS Noetic.

Options:

  -h              Display this help message and quit.

  -p PATH         Sets the absolute install location for the ROS workspace. If not specified,
                  the workspace directory will default to 'rl_ws'.

  -n              Install all packages and dependencies without prompting. This is useful if
                  you're running this script in a non-interactive terminal.

Examples:

  ./install_rl_env.sh -h
    This will display this help message and quit.

  ./install_rl_env.sh
    This will install the RL environment with the workspace directory defaulting to 'rl_ws'.
    It will prompt you to ask if you want to install certain packages and dependencies.

  ./install_rl_env.sh -p /path/to/workspace -n
    Skip prompts and install all packages and dependencies in the specified workspace path.
EOF
}

function failed() {
  echo -e "${ERR}[ERROR] $1${RRE}"
  exit 1
}

function prompt_user() {
  local prompt_message=$1
  local user_input
  if [ "$NONINTERACTIVE" = true ]; then
    user_input=""
  else
    echo -e "${BLU}${BOLD}$prompt_message${PROMPT}${NORM}${OFF}\c"
    read -r user_input
  fi
  echo $user_input
}

function check_or_create_workspace() {
  local ws_path=$1
  echo -e "${GRN}Checking for ROS workspace at $ws_path...${OFF}"
  if [ -d "$ws_path" ]; then
    echo -e "${GRN}Workspace found at $ws_path${OFF}"
  else
    echo -e "${RED}Workspace not found at $ws_path${OFF}"
    if [ "$NONINTERACTIVE" = true ]; then
      echo -e "${GRN}Creating workspace at $ws_path...${OFF}"
      mkdir -p "$ws_path/src" || failed "Failed to create workspace at $ws_path"
      echo -e "${GRN}Workspace created at $ws_path${OFF}"
    else
      local create_ws=$(prompt_user "Do you want to create a new workspace at $ws_path? (yes/no): ")
      if [[ $create_ws == [yY] || $create_ws == [yY][eE][sS] ]]; then
        mkdir -p "$ws_path/src" || failed "Failed to create workspace at $ws_path"
        echo -e "${GRN}Workspace created at $ws_path${OFF}"
      else
        ws_path=$(prompt_user "Please enter a valid workspace path or name: ")
        check_or_create_workspace "$ws_path"
      fi
    fi
  fi
}

function install_ros_noetic() {
  if ! dpkg-query -W -f='${Status}' ros-noetic-desktop-full 2>/dev/null | grep -c "ok installed" >/dev/null; then
    echo -e "${GRN}Installing ROS Noetic...${OFF}"
    sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' || failed "Failed to add ROS repository"
    sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 || failed "Failed to add ROS key"
    sudo apt update || failed "Failed to update package list"
    sudo apt install -y ros-noetic-desktop-full || failed "Failed to install ROS Noetic"
    echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
    source /opt/ros/noetic/setup.bash || failed "Failed to source ROS Noetic setup.bash"
    sudo apt install -y python3-rosdep || failed "Failed to install python3-rosdep"
    sudo rosdep init || failed "Failed to initialize rosdep"
    rosdep update || failed "Failed to update rosdep"
  else
    echo -e "${GRN}ROS Noetic is already installed!${OFF}"
  fi
}

function install_dependencies() {
  echo -e "${GRN}Installing dependencies...${OFF}"
  sudo apt install -y xterm ros-noetic-moveit python3-pykdl ros-noetic-urdf-parser-plugin ros-noetic-urdfdom-py ros-noetic-urdf-tutorial || failed "Failed to install dependencies"
}

function clone_and_build_packages() {
  local ws_path=$1
  echo -e "${GRN}Cloning and building packages in $ws_path...${OFF}"
  cd "$ws_path/src" || failed "Failed to navigate to $ws_path/src"

  git clone https://bitbucket.org/traclabs/trac_ik.git || failed "Failed to clone trac_ik"
  git clone https://github.com/ncbdrck/hrl-kdl.git || failed "Failed to clone hrl-kdl"

  cd hrl-kdl/pykdl_utils || failed "Failed to navigate to hrl-kdl/pykdl_utils"
  python3 setup.py build || failed "Failed to build hrl-kdl/pykdl_utils"
  sudo python3 setup.py install || failed "Failed to install hrl-kdl/pykdl_utils"

  cd ../hrl_geom || failed "Failed to navigate to hrl-kdl/hrl_geom"
  python3 setup.py build || failed "Failed to build hrl-kdl/hrl_geom"
  sudo python3 setup.py install || failed "Failed to install hrl-kdl/hrl_geom"

  cd "$ws_path/src" || failed "Failed to navigate to $ws_path/src"
  git clone --recurse-submodules -b gymnasium https://github.com/ncbdrck/uniros || failed "Failed to clone uniros"

  cd uniros || failed "Failed to navigate to uniros"
  git submodule update --remote --recursive || failed "Failed to update uniros submodules"

  sudo apt install -y python3-pip || failed "Failed to install python3-pip"

  pip3 install -r uniros/uniros/requirements.txt || failed "Failed to install uniros requirements"
  pip3 install -r uniros/multiros/requirements.txt || failed "Failed to install multiros requirements"
  pip3 install -r uniros/realros/requirements.txt || failed "Failed to install realros requirements"

  mkdir -p robot_packages/interbotix_robots
  cd robot_packages || failed "Failed to navigate to robot_packages"

  git clone https://github.com/NiryoRobotics/ned_ros || failed "Failed to clone ned_ros"
  git clone -b noetic-devel https://github.com/ros-industrial/universal_robot.git || failed "Failed to clone universal_robot"

  cd interbotix_robots || failed "Failed to navigate to interbotix_robots"
  git clone -b noetic https://github.com/Interbotix/interbotix_ros_core.git || failed "Failed to clone interbotix_ros_core"
  git clone -b noetic https://github.com/Interbotix/interbotix_ros_manipulators.git || failed "Failed to clone interbotix_ros_manipulators"
  git clone -b noetic https://github.com/Interbotix/interbotix_ros_toolboxes.git || failed "Failed to clone interbotix_ros_toolboxes"

  cd "$ws_path" || failed "Failed to navigate to $ws_path"
  rosdep install --from-paths src --ignore-src -r -y || failed "Failed to install rosdep dependencies"
  catkin build || failed "Failed to build the workspace"
  echo "source $ws_path/devel/setup.bash" >> ~/.bashrc
  source devel/setup.bash || failed "Failed to source workspace setup.bash"
}

# Main script starts here
while getopts 'hp:n' OPTION; do
  case "$OPTION" in
    h) help && exit 0;;
    p) WORKSPACE_PATH=$OPTARG;;
    n) NONINTERACTIVE=true;;
    *) echo "Unknown argument $OPTION" && exit 1;;
  esac
done

shift "$((OPTIND -1))"

if [ -z "$WORKSPACE_PATH" ]; then
  if [ "$NONINTERACTIVE" = true ]; then
    WORKSPACE_PATH="rl_ws"
  else
    WORKSPACE_PATH=$(prompt_user "Please enter the ROS workspace path (default is 'rl_ws'): ")
    if [ -z "$WORKSPACE_PATH" ]; then
      WORKSPACE_PATH="rl_ws"
    fi
  fi
fi

check_or_create_workspace "$WORKSPACE_PATH"
install_ros_noetic
install_dependencies
clone_and_build_packages "$WORKSPACE_PATH"

echo -e "${GRN}Installation and setup complete!${OFF}"
