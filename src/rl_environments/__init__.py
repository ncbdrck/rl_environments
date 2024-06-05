from gymnasium.envs.registration import register

# ---------------------------- Simulation Environments  ----------------------------

# ============================ RX200 Reacher Multiros Environments ============================

# ************************** RX200 Reacher Multiros Default Environments - Kinect **************************

# RX200 Reacher Multiros Default Environment
register(
    id='RX200ReacherSim-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_kinect_reach_sim:RX200ReacherEnv',
    max_episode_steps=1000,
)

# RX200 Reacher Multiros Default Environment - ee action space
register(
    id='RX200ReacherEESim-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_kinect_reach_sim:RX200ReacherEnv',
    max_episode_steps=1000,
    kwargs={'ee_action_type': True}
)

# RX200 Reacher Multiros default Environment with RGB Observation - using kinect v2
register(
    id='RX200kinectReacherSimRGB-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_kinect_reach_sim:RX200ReacherEnv',
    max_episode_steps=1000,
    kwargs={'rgb_obs_only': True, 'normal_obs_only': False}
)

# RX200 Reacher Multiros default Environment with RGB Observation - using kinect v2  - ee action space
register(
    id='RX200kinectReacherEESimRGB-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_kinect_reach_sim:RX200ReacherEnv',
    max_episode_steps=1000,
    kwargs={'rgb_obs_only': True, 'normal_obs_only': False, 'ee_action_type': True}
)


# RX200 Reacher Multiros default Environment with RGB and Normal Observation - using kinect v2
register(
    id='RX200kinectReacherSimRGBPlus-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_kinect_reach_sim:RX200ReacherEnv',
    max_episode_steps=1000,
    kwargs={'rgb_plus_normal_obs': True, 'normal_obs_only': False}
)

# RX200 Reacher Multiros default Environment with RGB and Normal Observation - using kinect v2  - ee action space
register(
    id='RX200kinectReacherEESimRGBPlus-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_kinect_reach_sim:RX200ReacherEnv',
    max_episode_steps=1000,
    kwargs={'rgb_plus_normal_obs': True, 'normal_obs_only': False, 'ee_action_type': True}
)

# RX200 Reacher Multiros default Environment with RGB, Depth and Normal Observation - using kinect v2
register(
    id='RX200kinectReacherSimRGBDepthPlus-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_kinect_reach_sim:RX200ReacherEnv',
    max_episode_steps=1000,
    kwargs={'rgb_plus_depth_plus_normal_obs': True, 'normal_obs_only': False}
)

# RX200 Reacher Multiros default Environment with RGB, Depth and Normal Observation - using kinect v2  - ee action space
register(
    id='RX200kinectReacherEESimRGBDepthPlus-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_kinect_reach_sim:RX200ReacherEnv',
    max_episode_steps=1000,
    kwargs={'rgb_plus_depth_plus_normal_obs': True, 'normal_obs_only': False, 'ee_action_type': True}
)




# ************************** RX200 Reacher Multiros Goal Environments - Kinect **************************

# RX200 Reacher Multiros Goal Environment
register(
    id='RX200ReacherGoalSim-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_kinect_reach_goal_sim:RX200ReacherGoalEnv',
    max_episode_steps=1000,
)

# RX200 Reacher Multiros Goal Environment - ee action space
register(
    id='RX200ReacherEEGoalSim-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_kinect_reach_goal_sim:RX200ReacherGoalEnv',
    max_episode_steps=1000,
    kwargs={'ee_action_type': True}
)

# RX200 Reacher Multiros Goal Environment with RGB Observation - using kinect v2
register(
    id='RX200kinectReacherGoalSimRGB-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_kinect_reach_goal_sim:RX200ReacherGoalEnv',
    max_episode_steps=1000,
    kwargs={'rgb_obs_only': True, 'normal_obs_only': False}
)

# RX200 Reacher Multiros Goal Environment with RGB Observation - using kinect v2  - ee action space
register(
    id='RX200kinectReacherEEGoalSimRGB-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_kinect_reach_goal_sim:RX200ReacherGoalEnv',
    max_episode_steps=1000,
    kwargs={'rgb_obs_only': True, 'normal_obs_only': False, 'ee_action_type': True}
)

# RX200 Reacher Multiros Goal Environment with RGB and Normal Observation - using kinect v2
register(
    id='RX200kinectReacherGoalSimRGBPlus-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_kinect_reach_goal_sim:RX200ReacherGoalEnv',
    max_episode_steps=1000,
    kwargs={'rgb_plus_normal_obs': True, 'normal_obs_only': False}
)

# RX200 Reacher Multiros Goal Environment with RGB and Normal Observation - using kinect v2  - ee action space
register(
    id='RX200kinectReacherEEGoalSimRGBPlus-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_kinect_reach_goal_sim:RX200ReacherGoalEnv',
    max_episode_steps=1000,
    kwargs={'rgb_plus_normal_obs': True, 'normal_obs_only': False, 'ee_action_type': True}
)

# RX200 Reacher Multiros Goal Environment with RGB, Depth and Normal Observation - using kinect v2
register(
    id='RX200kinectReacherGoalSimRGBDepthPlus-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_kinect_reach_goal_sim:RX200ReacherGoalEnv',
    max_episode_steps=1000,
    kwargs={'rgb_plus_depth_plus_normal_obs': True, 'normal_obs_only': False}
)

# RX200 Reacher Multiros Goal Environment with RGB, Depth and Normal Observation - using kinect v2  - ee action space
register(
    id='RX200kinectReacherEEGoalSimRGBDepthPlus-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_kinect_reach_goal_sim:RX200ReacherGoalEnv',
    max_episode_steps=1000,
    kwargs={'rgb_plus_depth_plus_normal_obs': True, 'normal_obs_only': False, 'ee_action_type': True}
)





# ************************** RX200 Reacher Multiros Default Environments - ZED2 **************************

# RX200 Reacher Multiros Default Environment with RGB Observation - using ZED2
register(
    id='RX200Zed2ReacherSimRGB-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_zed2_reach_sim:RX200ReacherEnv',
    max_episode_steps=1000,
    kwargs={'rgb_obs_only': True, 'normal_obs_only': False}
)

# RX200 Reacher Multiros Default Environment with RGB Observation - using ZED2  - ee action space
register(
    id='RX200Zed2ReacherEESimRGB-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_zed2_reach_sim:RX200ReacherEnv',
    max_episode_steps=1000,
    kwargs={'rgb_obs_only': True, 'normal_obs_only': False, 'ee_action_type': True}
)

# RX200 Reacher Multiros default Environment with RGB and Normal Observation - using ZED2
register(
    id='RX200Zed2ReacherSimRGBPlus-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_zed2_reach_sim:RX200ReacherEnv',
    max_episode_steps=1000,
    kwargs={'rgb_plus_normal_obs': True, 'normal_obs_only': False}
)

# RX200 Reacher Multiros default Environment with RGB and Normal Observation - using ZED2  - ee action space
register(
    id='RX200Zed2ReacherEESimRGBPlus-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_zed2_reach_sim:RX200ReacherEnv',
    max_episode_steps=1000,
    kwargs={'rgb_plus_normal_obs': True, 'normal_obs_only': False, 'ee_action_type': True}
)

# RX200 Reacher Multiros default Environment with RGB, Depth and Normal Observation - using ZED2
register(
    id='RX200Zed2ReacherSimRGBDepthPlus-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_zed2_reach_sim:RX200ReacherEnv',
    max_episode_steps=1000,
    kwargs={'rgb_plus_depth_plus_normal_obs': True, 'normal_obs_only': False}
)

# RX200 Reacher Multiros default Environment with RGB, Depth and Normal Observation - using ZED2  - ee action space
register(
    id='RX200Zed2ReacherEESimRGBDepthPlus-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_zed2_reach_sim:RX200ReacherEnv',
    max_episode_steps=1000,
    kwargs={'rgb_plus_depth_plus_normal_obs': True, 'normal_obs_only': False, 'ee_action_type': True}
)





# ************************** RX200 Reacher Multiros Goal Environments - ZED2 **************************

# RX200 Reacher Multiros Goal Environment with RGB Observation - using ZED2
register(
    id='RX200Zed2ReacherGoalSimRGB-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_zed2_reach_goal_sim:RX200ReacherGoalEnv',
    max_episode_steps=1000,
    kwargs={'rgb_obs_only': True, 'normal_obs_only': False}
)

# RX200 Reacher Multiros Goal Environment with RGB Observation - using ZED2  - ee action space
register(
    id='RX200Zed2ReacherEEGoalSimRGB-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_zed2_reach_goal_sim:RX200ReacherGoalEnv',
    max_episode_steps=1000,
    kwargs={'rgb_obs_only': True, 'normal_obs_only': False, 'ee_action_type': True}
)

# RX200 Reacher Multiros Goal Environment with RGB and Normal Observation - using ZED2
register(
    id='RX200Zed2ReacherGoalSimRGBPlus-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_zed2_reach_goal_sim:RX200ReacherGoalEnv',
    max_episode_steps=1000,
    kwargs={'rgb_plus_normal_obs': True, 'normal_obs_only': False}
)

# RX200 Reacher Multiros Goal Environment with RGB and Normal Observation - using ZED2  - ee action space
register(
    id='RX200Zed2ReacherEEGoalSimRGBPlus-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_zed2_reach_goal_sim:RX200ReacherGoalEnv',
    max_episode_steps=1000,
    kwargs={'rgb_plus_normal_obs': True, 'normal_obs_only': False, 'ee_action_type': True}
)

# RX200 Reacher Multiros Goal Environment with RGB, Depth and Normal Observation - using ZED2
register(
    id='RX200Zed2ReacherGoalSimRGBDepthPlus-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_zed2_reach_goal_sim:RX200ReacherGoalEnv',
    max_episode_steps=1000,
    kwargs={'rgb_plus_depth_plus_normal_obs': True, 'normal_obs_only': False}
)

# RX200 Reacher Multiros Goal Environment with RGB, Depth and Normal Observation - using ZED2  - ee action space
register(
    id='RX200Zed2ReacherEEGoalSimRGBDepthPlus-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_zed2_reach_goal_sim:RX200ReacherGoalEnv',
    max_episode_steps=1000,
    kwargs={'rgb_plus_depth_plus_normal_obs': True, 'normal_obs_only': False, 'ee_action_type': True}
)




