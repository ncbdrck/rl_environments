from gymnasium.envs.registration import register

ALL_REACH_SIM_NAMES = [
    'RX200ReacherSim-v0',  # RX200 Reacher Multiros Default Environment - v0
    'RX200ReacherSim-v1',  # RX200 Reacher Multiros Default Environment - v1
    'RX200ReacherSim-v2',  # RX200 Reacher Multiros Default Environment - v2
    'RX200ReacherEESim-v0',  # RX200 Reacher Multiros Default Environment - ee action space
    'RX200kinectReacherSimRGB-v0',  # RX200 Reacher Multiros Default Environment with RGB Observation - using kinect v2
    'RX200kinectReacherEESimRGB-v0',
    # RX200 Reacher Multiros Default Environment with RGB Observation - using kinect v2 - ee action space
    'RX200kinectReacherSimRGBPlus-v0',
    # RX200 Reacher Multiros Default Environment with RGB and Normal Observation - using kinect v2
    'RX200kinectReacherEESimRGBPlus-v0',
    # RX200 Reacher Multiros Default Environment with RGB and Normal Observation - using kinect v2 - ee action space
    'RX200kinectReacherSimRGBDepthPlus-v0',
    # RX200 Reacher Multiros Default Environment with RGB, Depth and Normal Observation - using kinect v2
    'RX200kinectReacherEESimRGBDepthPlus-v0',
    # RX200 Reacher Multiros Default Environment with RGB, Depth and Normal Observation - using kinect v2 - ee action space
    'RX200ReacherGoalSim-v0',  # RX200 Reacher Multiros Goal Environment
    'RX200ReacherEEGoalSim-v0',  # RX200 Reacher Multiros Goal Environment - ee action space
    'RX200kinectReacherGoalSimRGB-v0',  # RX200 Reacher Multiros Goal Environment with RGB Observation - using kinect v2
    'RX200kinectReacherEEGoalSimRGB-v0',
    # RX200 Reacher Multiros Goal Environment with RGB Observation - using kinect v2 - ee action space
    'RX200kinectReacherGoalSimRGBPlus-v0',
    # RX200 Reacher Multiros Goal Environment with RGB and Normal Observation - using kinect v2
    'RX200kinectReacherEEGoalSimRGBPlus-v0',
    # RX200 Reacher Multiros Goal Environment with RGB and Normal Observation - using kinect v2 - ee action space
    'RX200kinectReacherGoalSimRGBDepthPlus-v0',
    # RX200 Reacher Multiros Goal Environment with RGB, Depth and Normal Observation - using kinect v2
    'RX200kinectReacherEEGoalSimRGBDepthPlus-v0',
    # RX200 Reacher Multiros Goal Environment with RGB, Depth and Normal Observation - using kinect v2 - ee action space
    'RX200Zed2ReacherSimRGB-v0',  # RX200 Reacher Multiros Default Environment with RGB Observation - using ZED2
    'RX200Zed2ReacherEESimRGB-v0',
    # RX200 Reacher Multiros Default Environment with RGB Observation - using ZED2 - ee action space
    'RX200Zed2ReacherSimRGBPlus-v0',
    # RX200 Reacher Multiros default Environment with RGB and Normal Observation - using ZED2
    'RX200Zed2ReacherEESimRGBPlus-v0',
    # RX200 Reacher Multiros default Environment with RGB and Normal Observation - using ZED2 - ee action space
    'RX200Zed2ReacherSimRGBDepthPlus-v0',
    # RX200 Reacher Multiros default Environment with RGB, Depth and Normal Observation - using ZED2
    'RX200Zed2ReacherEESimRGBDepthPlus-v0',
    # RX200 Reacher Multiros default Environment with RGB, Depth and Normal Observation - using ZED2 - ee action space
    'RX200Zed2ReacherGoalSimRGB-v0',  # RX200 Reacher Multiros Goal Environment with RGB Observation - using ZED2
    'RX200Zed2ReacherEEGoalSimRGB-v0',
    # RX200 Reacher Multiros Goal Environment with RGB Observation - using ZED2 - ee action space
    'RX200Zed2ReacherGoalSimRGBPlus-v0',
    # RX200 Reacher Multiros Goal Environment with RGB and Normal Observation - using ZED2
    'RX200Zed2ReacherEEGoalSimRGBPlus-v0',
    # RX200 Reacher Multiros Goal Environment with RGB and Normal Observation - using ZED2 - ee action space
    'RX200Zed2ReacherGoalSimRGBDepthPlus-v0',
    # RX200 Reacher Multiros Goal Environment with RGB, Depth and Normal Observation - using ZED2
    'RX200Zed2ReacherEEGoalSimRGBDepthPlus-v0',
    # RX200 Reacher Multiros Goal Environment with RGB, Depth and Normal Observation - using ZED2 - ee action space
]

ALL_PUSH_SIM_NAMES = [
    'RX200PushSim-v0',  # RX200 Push Multiros Default Environment
    'RX200PushSim-v1',  # RX200 Push Multiros Default Environment - v1
    'RX200PushEESim-v0',  # RX200 Push Multiros Default Environment - ee action space
    'RX200kinectPushSimRGB-v0',  # RX200 Push Multiros Default Environment with RGB Observation - using kinect v2
    'RX200kinectPushEESimRGB-v0',
    # RX200 Push Multiros Default Environment with RGB Observation - using kinect v2 - ee action space
    'RX200kinectPushSimRGBPlus-v0',
    # RX200 Push Multiros default Environment with RGB and Normal Observation - using kinect v2
    'RX200kinectPushEESimRGBPlus-v0',
    # RX200 Push Multiros default Environment with RGB and Normal Observation - using kinect v2 - ee action space
    'RX200kinectPushSimRGBDepthPlus-v0',
    # RX200 Push Multiros default Environment with RGB, Depth and Normal Observation - using kinect v2
    'RX200kinectPushEESimRGBDepthPlus-v0',
    # RX200 Push Multiros default Environment with RGB, Depth and Normal Observation - using kinect v2 - ee action space
]

# ---------------------------- Simulation Environments  ----------------------------

# ============================ RX200 Reacher Multiros Environments ============================

# ************************** RX200 Reacher Multiros Default Environments - Kinect **************************

# RX200 Reacher Multiros Default Environment
register(
    id='RX200ReacherSim-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_kinect_reach_sim:RX200ReacherEnv',
    max_episode_steps=100,
)

# RX200 Reacher Multiros Default Environment
register(
    id='RX200ReacherSim-v1',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_kinect_reach_sim_v1:RX200ReacherEnv',
    max_episode_steps=100,
)

# RX200 Reacher Multiros Default Environment
register(
    id='RX200ReacherSim-v2',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_kinect_reach_sim_v2:RX200ReacherEnv',
    max_episode_steps=100,
)

# RX200 Reacher Multiros Default Environment - ee action space
register(
    id='RX200ReacherEESim-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_kinect_reach_sim:RX200ReacherEnv',
    max_episode_steps=100,
    kwargs={'ee_action_type': True}
)

# RX200 Reacher Multiros default Environment with RGB Observation - using kinect v2
register(
    id='RX200kinectReacherSimRGB-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_kinect_reach_sim:RX200ReacherEnv',
    max_episode_steps=100,
    kwargs={'rgb_obs_only': True, 'normal_obs_only': False}
)

# RX200 Reacher Multiros default Environment with RGB Observation - using kinect v2 - ee action space
register(
    id='RX200kinectReacherEESimRGB-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_kinect_reach_sim:RX200ReacherEnv',
    max_episode_steps=100,
    kwargs={'rgb_obs_only': True, 'normal_obs_only': False, 'ee_action_type': True}
)

# RX200 Reacher Multiros default Environment with RGB and Normal Observation - using kinect v2
register(
    id='RX200kinectReacherSimRGBPlus-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_kinect_reach_sim:RX200ReacherEnv',
    max_episode_steps=100,
    kwargs={'rgb_plus_normal_obs': True, 'normal_obs_only': False}
)

# RX200 Reacher Multiros default Environment with RGB and Normal Observation - using kinect v2 - ee action space
register(
    id='RX200kinectReacherEESimRGBPlus-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_kinect_reach_sim:RX200ReacherEnv',
    max_episode_steps=100,
    kwargs={'rgb_plus_normal_obs': True, 'normal_obs_only': False, 'ee_action_type': True}
)

# RX200 Reacher Multiros default Environment with RGB, Depth and Normal Observation - using kinect v2
register(
    id='RX200kinectReacherSimRGBDepthPlus-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_kinect_reach_sim:RX200ReacherEnv',
    max_episode_steps=100,
    kwargs={'rgb_plus_depth_plus_normal_obs': True, 'normal_obs_only': False}
)

# RX200 Reacher Multiros default Environment with RGB, Depth and Normal Observation - using kinect v2 - ee action space
register(
    id='RX200kinectReacherEESimRGBDepthPlus-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_kinect_reach_sim:RX200ReacherEnv',
    max_episode_steps=100,
    kwargs={'rgb_plus_depth_plus_normal_obs': True, 'normal_obs_only': False, 'ee_action_type': True}
)

# ************************** RX200 Reacher Multiros Goal Environments - Kinect **************************

# RX200 Reacher Multiros Goal Environment
register(
    id='RX200ReacherGoalSim-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_kinect_reach_goal_sim:RX200ReacherGoalEnv',
    max_episode_steps=100,
)

# RX200 Reacher Multiros Goal Environment - ee action space
register(
    id='RX200ReacherEEGoalSim-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_kinect_reach_goal_sim:RX200ReacherGoalEnv',
    max_episode_steps=100,
    kwargs={'ee_action_type': True}
)

# RX200 Reacher Multiros Goal Environment with RGB Observation - using kinect v2
register(
    id='RX200kinectReacherGoalSimRGB-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_kinect_reach_goal_sim:RX200ReacherGoalEnv',
    max_episode_steps=100,
    kwargs={'rgb_obs_only': True, 'normal_obs_only': False}
)

# RX200 Reacher Multiros Goal Environment with RGB Observation - using kinect v2 - ee action space
register(
    id='RX200kinectReacherEEGoalSimRGB-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_kinect_reach_goal_sim:RX200ReacherGoalEnv',
    max_episode_steps=100,
    kwargs={'rgb_obs_only': True, 'normal_obs_only': False, 'ee_action_type': True}
)

# RX200 Reacher Multiros Goal Environment with RGB and Normal Observation - using kinect v2
register(
    id='RX200kinectReacherGoalSimRGBPlus-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_kinect_reach_goal_sim:RX200ReacherGoalEnv',
    max_episode_steps=100,
    kwargs={'rgb_plus_normal_obs': True, 'normal_obs_only': False}
)

# RX200 Reacher Multiros Goal Environment with RGB and Normal Observation - using kinect v2 - ee action space
register(
    id='RX200kinectReacherEEGoalSimRGBPlus-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_kinect_reach_goal_sim:RX200ReacherGoalEnv',
    max_episode_steps=100,
    kwargs={'rgb_plus_normal_obs': True, 'normal_obs_only': False, 'ee_action_type': True}
)

# RX200 Reacher Multiros Goal Environment with RGB, Depth and Normal Observation - using kinect v2
register(
    id='RX200kinectReacherGoalSimRGBDepthPlus-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_kinect_reach_goal_sim:RX200ReacherGoalEnv',
    max_episode_steps=100,
    kwargs={'rgb_plus_depth_plus_normal_obs': True, 'normal_obs_only': False}
)

# RX200 Reacher Multiros Goal Environment with RGB, Depth and Normal Observation - using kinect v2 - ee action space
register(
    id='RX200kinectReacherEEGoalSimRGBDepthPlus-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_kinect_reach_goal_sim:RX200ReacherGoalEnv',
    max_episode_steps=100,
    kwargs={'rgb_plus_depth_plus_normal_obs': True, 'normal_obs_only': False, 'ee_action_type': True}
)

# ************************** RX200 Reacher Multiros Default Environments - ZED2 **************************

# RX200 Reacher Multiros Default Environment with RGB Observation - using ZED2
register(
    id='RX200Zed2ReacherSimRGB-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_zed2_reach_sim:RX200ReacherEnv',
    max_episode_steps=100,
    kwargs={'rgb_obs_only': True, 'normal_obs_only': False}
)

# RX200 Reacher Multiros Default Environment with RGB Observation - using ZED2 - ee action space
register(
    id='RX200Zed2ReacherEESimRGB-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_zed2_reach_sim:RX200ReacherEnv',
    max_episode_steps=100,
    kwargs={'rgb_obs_only': True, 'normal_obs_only': False, 'ee_action_type': True}
)

# RX200 Reacher Multiros default Environment with RGB and Normal Observation - using ZED2
register(
    id='RX200Zed2ReacherSimRGBPlus-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_zed2_reach_sim:RX200ReacherEnv',
    max_episode_steps=100,
    kwargs={'rgb_plus_normal_obs': True, 'normal_obs_only': False}
)

# RX200 Reacher Multiros default Environment with RGB and Normal Observation - using ZED2 - ee action space
register(
    id='RX200Zed2ReacherEESimRGBPlus-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_zed2_reach_sim:RX200ReacherEnv',
    max_episode_steps=100,
    kwargs={'rgb_plus_normal_obs': True, 'normal_obs_only': False, 'ee_action_type': True}
)

# RX200 Reacher Multiros default Environment with RGB, Depth and Normal Observation - using ZED2
register(
    id='RX200Zed2ReacherSimRGBDepthPlus-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_zed2_reach_sim:RX200ReacherEnv',
    max_episode_steps=100,
    kwargs={'rgb_plus_depth_plus_normal_obs': True, 'normal_obs_only': False}
)

# RX200 Reacher Multiros default Environment with RGB, Depth and Normal Observation - using ZED2 - ee action space
register(
    id='RX200Zed2ReacherEESimRGBDepthPlus-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_zed2_reach_sim:RX200ReacherEnv',
    max_episode_steps=100,
    kwargs={'rgb_plus_depth_plus_normal_obs': True, 'normal_obs_only': False, 'ee_action_type': True}
)

# ************************** RX200 Reacher Multiros Goal Environments - ZED2 **************************

# RX200 Reacher Multiros Goal Environment with RGB Observation - using ZED2
register(
    id='RX200Zed2ReacherGoalSimRGB-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_zed2_reach_goal_sim:RX200ReacherGoalEnv',
    max_episode_steps=100,
    kwargs={'rgb_obs_only': True, 'normal_obs_only': False}
)

# RX200 Reacher Multiros Goal Environment with RGB Observation - using ZED2 - ee action space
register(
    id='RX200Zed2ReacherEEGoalSimRGB-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_zed2_reach_goal_sim:RX200ReacherGoalEnv',
    max_episode_steps=100,
    kwargs={'rgb_obs_only': True, 'normal_obs_only': False, 'ee_action_type': True}
)

# RX200 Reacher Multiros Goal Environment with RGB and Normal Observation - using ZED2
register(
    id='RX200Zed2ReacherGoalSimRGBPlus-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_zed2_reach_goal_sim:RX200ReacherGoalEnv',
    max_episode_steps=100,
    kwargs={'rgb_plus_normal_obs': True, 'normal_obs_only': False}
)

# RX200 Reacher Multiros Goal Environment with RGB and Normal Observation - using ZED2 - ee action space
register(
    id='RX200Zed2ReacherEEGoalSimRGBPlus-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_zed2_reach_goal_sim:RX200ReacherGoalEnv',
    max_episode_steps=100,
    kwargs={'rgb_plus_normal_obs': True, 'normal_obs_only': False, 'ee_action_type': True}
)

# RX200 Reacher Multiros Goal Environment with RGB, Depth and Normal Observation - using ZED2
register(
    id='RX200Zed2ReacherGoalSimRGBDepthPlus-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_zed2_reach_goal_sim:RX200ReacherGoalEnv',
    max_episode_steps=100,
    kwargs={'rgb_plus_depth_plus_normal_obs': True, 'normal_obs_only': False}
)

# RX200 Reacher Multiros Goal Environment with RGB, Depth and Normal Observation - using ZED2 - ee action space
register(
    id='RX200Zed2ReacherEEGoalSimRGBDepthPlus-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_zed2_reach_goal_sim:RX200ReacherGoalEnv',
    max_episode_steps=100,
    kwargs={'rgb_plus_depth_plus_normal_obs': True, 'normal_obs_only': False, 'ee_action_type': True}
)

# ============================ RX200 Push Multiros Environments ============================
# ************************** RX200 Push Multiros Default Environments - Kinect **************************

# RX200 Push Multiros Default Environment
register(
    id='RX200PushSim-v0',
    entry_point='rl_environments.rx200.sim.task_envs.push.rx200_kinect_push_sim:RX200PushEnv',
    max_episode_steps=100,
)

register(
    id='RX200PushSim-v1',
    entry_point='rl_environments.rx200.sim.task_envs.push.rx200_kinect_push_sim_v1:RX200PushEnv',
    max_episode_steps=100,
)

# RX200 Push Multiros Default Environment - ee action space
register(
    id='RX200PushEESim-v0',
    entry_point='rl_environments.rx200.sim.task_envs.push.rx200_kinect_push_sim:RX200PushEnv',
    max_episode_steps=100,
    kwargs={'ee_action_type': True}
)

# RX200 Push Multiros default Environment with RGB Observation - using kinect v2
register(
    id='RX200kinectPushSimRGB-v0',
    entry_point='rl_environments.rx200.sim.task_envs.push.rx200_kinect_push_sim:RX200PushEnv',
    max_episode_steps=100,
    kwargs={'rgb_obs_only': True, 'normal_obs_only': False}
)

# RX200 Push Multiros default Environment with RGB Observation - using kinect v2 - ee action space
register(
    id='RX200kinectPushEESimRGB-v0',
    entry_point='rl_environments.rx200.sim.task_envs.push.rx200_kinect_push_sim:RX200PushEnv',
    max_episode_steps=100,
    kwargs={'rgb_obs_only': True, 'normal_obs_only': False, 'ee_action_type': True}
)

# RX200 Push Multiros default Environment with RGB and Normal Observation - using kinect v2
register(
    id='RX200kinectPushSimRGBPlus-v0',
    entry_point='rl_environments.rx200.sim.task_envs.push.rx200_kinect_push_sim:RX200PushEnv',
    max_episode_steps=100,
    kwargs={'rgb_plus_normal_obs': True, 'normal_obs_only': False}
)

# RX200 Push Multiros default Environment with RGB and Normal Observation - using kinect v2 - ee action space
register(
    id='RX200kinectPushEESimRGBPlus-v0',
    entry_point='rl_environments.rx200.sim.task_envs.push.rx200_kinect_push_sim:RX200PushEnv',
    max_episode_steps=100,
    kwargs={'rgb_plus_normal_obs': True, 'normal_obs_only': False, 'ee_action_type': True}
)

# RX200 Push Multiros default Environment with RGB, Depth and Normal Observation - using kinect v2
register(
    id='RX200kinectPushSimRGBDepthPlus-v0',
    entry_point='rl_environments.rx200.sim.task_envs.push.rx200_kinect_push_sim:RX200PushEnv',
    max_episode_steps=100,
    kwargs={'rgb_plus_depth_plus_normal_obs': True, 'normal_obs_only': False}
)

# RX200 Push Multiros default Environment with RGB, Depth and Normal Observation - using kinect v2 - ee action space
register(
    id='RX200kinectPushEESimRGBDepthPlus-v0',
    entry_point='rl_environments.rx200.sim.task_envs.push.rx200_kinect_push_sim:RX200PushEnv',
    max_episode_steps=100,
    kwargs={'rgb_plus_depth_plus_normal_obs': True, 'normal_obs_only': False, 'ee_action_type': True}
)

# ---------------------------- Real Environments  ----------------------------
# ============================ RX200 Reacher RealROS Environments ============================
# RX200 Reacher RealROS Environment - Default
register(
    id='RX200ReacherReal-v0',
    entry_point='rl_environments.rx200.real.task_envs.reach.rx200_reach_real:RX200ReacherEnv',
    max_episode_steps=100,
)

# RX200 Reacher RealROS Environment - Goal Conditioned
register(
    id='RX200ReacherGoalReal-v0',
    entry_point='rl_environments.rx200.real.task_envs.reach.rx200_reach_goal_real:RX200ReacherGoalEnv',
    max_episode_steps=100,
)
