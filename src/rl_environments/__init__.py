from gymnasium.envs.registration import register

# ---------------------------- Simulation Environments  ----------------------------

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

# RX200 Reacher Multiros Goal Environment
register(
    id='RX200ReacherGoalSim-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_kinect_reach_goal_sim:RX200ReacherGoalEnv',
    max_episode_steps=1000,
)
