from gymnasium.envs.registration import register

# ---------------------------- Simulation Environments  ----------------------------

# RX200 Reacher Multiros Default Environment
register(
    id='RX200ReacherSim-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_reach_sim:RX200ReacherEnv',
    max_episode_steps=1000,
)

# RX200 Reacher Multiros Goal Environment
register(
    id='RX200ReacherGoalSim-v0',
    entry_point='rl_environments.rx200.sim.task_envs.reach.rx200_reach_goal_sim:RX200ReacherGoalEnv',
    max_episode_steps=1000,
)