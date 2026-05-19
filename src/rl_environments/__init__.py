"""
Gymnasium registrations for the rl_environments package.

Convention: **one id per task env file**. Configuration variants (RGB vs
depth observations, joint vs EE action space, vision sensor combos) are
exposed as ``__init__`` kwargs on the task env classes — pass them at
construction time::

    env = gym.make("RX200ReacherSim-v3",
                   gazebo_gui=True,
                   ee_action_type=True,
                   rgb_obs_only=True, normal_obs_only=False,
                   reward_type="Sparse")

Earlier the registry held ~46 ids: same files registered repeatedly with
different ``kwargs={...}`` to expose each combo. Slimmed down 2026-05-17:
all kwargs-only aliases dropped (33 ids removed); the remaining 13 ids
each point at a distinct task env file.
"""
from gymnasium.envs.registration import register


ALL_REACH_SIM_NAMES = [
    # Kinect-based RX200 reach — four design iterations preserved as
    # separate ids. v3 is the latest (per-link FK safety + close-race
    # guard). v0/v1/v2 kept for historical comparison.
    "RX200ReacherSim-v0",
    "RX200ReacherSim-v1",
    "RX200ReacherSim-v2",
    "RX200ReacherSim-v3",
    "RX200ReacherGoalSim-v0",
    # ZED2-based RX200 reach.
    "RX200Zed2ReacherSim-v0",
    "RX200Zed2ReacherGoalSim-v0",
    # Ned2 reach.
    "NED2ReacherSim-v0",
    "NED2ReacherGoalSim-v0",
]

ALL_PUSH_SIM_NAMES = [
    "RX200PushSim-v0",
    "RX200PushSim-v1",  # latest (per-link FK safety + close-race guard)
]

ALL_REACH_REAL_NAMES = [
    "RX200ReacherReal-v0",
    "RX200ReacherGoalReal-v0",
]


# ---------------------------- Simulation Environments  ----------------------------

# ============================ RX200 Reacher Multiros Environments ============================

# Kinect-based RX200 reach — historical iterations v0/v1/v2 kept registered
# for side-by-side comparison; v3 is the polished, per-link-FK-safe variant.
register(
    id="RX200ReacherSim-v0",
    entry_point="rl_environments.rx200.sim.task_envs.reach.rx200_kinect_reach_sim:RX200ReacherEnv",
    max_episode_steps=100,
)
register(
    id="RX200ReacherSim-v1",
    entry_point="rl_environments.rx200.sim.task_envs.reach.rx200_kinect_reach_sim_v1:RX200ReacherEnv",
    max_episode_steps=100,
)
register(
    id="RX200ReacherSim-v2",
    entry_point="rl_environments.rx200.sim.task_envs.reach.rx200_kinect_reach_sim_v2:RX200ReacherEnv",
    max_episode_steps=100,
)
# v3 adds per-link FK safety in execute_action (via
# RX200RobotEnv._check_action_links_safe) so the arm can't fold down with
# shoulder/elbow/wrist below the table even when EE stays above. Also
# reads new safety rosparams from rx200_reach_task_config.yaml
# (table_z, safety_z_margin[_real], max_joint_delta[_real]).
register(
    id="RX200ReacherSim-v3",
    entry_point="rl_environments.rx200.sim.task_envs.reach.rx200_kinect_reach_sim_v3:RX200ReacherEnv",
    max_episode_steps=100,
)

# Kinect-based RX200 reach, goal-conditioned (Dict obs space).
register(
    id="RX200ReacherGoalSim-v0",
    entry_point="rl_environments.rx200.sim.task_envs.reach.rx200_kinect_reach_goal_sim:RX200ReacherGoalEnv",
    max_episode_steps=100,
)

# ZED2-based RX200 reach (different vision sensor than kinect).
register(
    id="RX200Zed2ReacherSim-v0",
    entry_point="rl_environments.rx200.sim.task_envs.reach.rx200_zed2_reach_sim:RX200ReacherEnv",
    max_episode_steps=100,
)
register(
    id="RX200Zed2ReacherGoalSim-v0",
    entry_point="rl_environments.rx200.sim.task_envs.reach.rx200_zed2_reach_goal_sim:RX200ReacherGoalEnv",
    max_episode_steps=100,
)


# ============================ RX200 Push Multiros Environments ============================

# v0 and v1 are separate design iterations. v1 has per-link FK safety +
# close-race guard wired in (a2f7e9b).
register(
    id="RX200PushSim-v0",
    entry_point="rl_environments.rx200.sim.task_envs.push.rx200_kinect_push_sim:RX200PushEnv",
    max_episode_steps=100,
)
register(
    id="RX200PushSim-v1",
    entry_point="rl_environments.rx200.sim.task_envs.push.rx200_kinect_push_sim_v1:RX200PushEnv",
    max_episode_steps=100,
)


# ============================ Ned2 Reacher Multiros Environments ============================

register(
    id="NED2ReacherSim-v0",
    entry_point="rl_environments.ned2.sim.task_envs.reach.ned2_reach_sim:NED2ReacherEnv",
    max_episode_steps=100,
)
register(
    id="NED2ReacherGoalSim-v0",
    entry_point="rl_environments.ned2.sim.task_envs.reach.ned2_reach_goal_sim:NED2ReacherGoalEnv",
    max_episode_steps=100,
)


# ---------------------------- Real Environments  ----------------------------

# ============================ RX200 Reacher RealROS Environments ============================
register(
    id="RX200ReacherReal-v0",
    entry_point="rl_environments.rx200.real.task_envs.reach.rx200_reach_real:RX200ReacherEnv",
    max_episode_steps=100,
)
register(
    id="RX200ReacherGoalReal-v0",
    entry_point="rl_environments.rx200.real.task_envs.reach.rx200_reach_goal_real:RX200ReacherGoalEnv",
    max_episode_steps=100,
)
