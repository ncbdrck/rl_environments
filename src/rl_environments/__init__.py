"""
Gymnasium registrations for the rl_environments package.

Convention: **one id per task env file**. Configuration variants (RGB vs
depth observations, joint vs EE action space, vision sensor combos,
real-time vs MDP pause-step-resume mode) are exposed as ``__init__``
kwargs on the task env classes — pass them at construction time::

    env = gym.make("RX200ReacherSim-v0",
                   gazebo_gui=True,
                   ee_action_type=True,
                   rgb_obs_only=True, normal_obs_only=False,
                   reward_type="Sparse",
                   realtime_mode=True)

Iteration history: 2026-05-17 slim dropped 33 kwargs-only aliases (46 →
13 ids). 2026-05-19 collapsed RX200 reach v0/v1/v2/v3 → single v0 and
push v0/v1 → single v0; slide envs removed entirely. The polished
per-link-FK-safe + close-race-guarded code (formerly v3 reach / v1
push) is now the canonical v0.
"""
from gymnasium.envs.registration import register


ALL_REACH_SIM_NAMES = [
    # Kinect-based RX200 reach.
    "RX200ReacherSim-v0",
    "RX200ReacherGoalSim-v0",
    # ZED2-based RX200 reach.
    "RX200Zed2ReacherSim-v0",
    "RX200Zed2ReacherGoalSim-v0",
    # Ned2 reach.
    "NED2ReacherSim-v0",
    "NED2ReacherGoalSim-v0",
    # ViperX-300 S reach.
    "VX300SReacherSim-v0",
    "VX300SReacherGoalSim-v0",
]

ALL_PUSH_SIM_NAMES = [
    "RX200PushSim-v0",
    "RX200PushGoalSim-v0",
    "RX200Zed2PushSim-v0",
    "RX200Zed2PushGoalSim-v0",
    # Ned2 push.
    "NED2PushSim-v0",
    "NED2PushGoalSim-v0",
]

ALL_PNP_SIM_NAMES = [
    "RX200PnPSim-v0",
    "RX200PnPGoalSim-v0",
    "RX200Zed2PnPSim-v0",
    "RX200Zed2PnPGoalSim-v0",
    # Ned2 PnP.
    "NED2PnPSim-v0",
    "NED2PnPGoalSim-v0",
]

ALL_REACH_REAL_NAMES = [
    "RX200ReacherReal-v0",
    "RX200ReacherGoalReal-v0",
    # Ned2 reach real.
    "NED2ReacherReal-v0",
    "NED2ReacherGoalReal-v0",
    # ViperX-300 S reach real.
    "VX300SReacherReal-v0",
    "VX300SReacherGoalReal-v0",
]

ALL_PUSH_REAL_NAMES = [
    "RX200PushReal-v0",
    "RX200PushGoalReal-v0",
    # Ned2 push real.
    "NED2PushReal-v0",
    "NED2PushGoalReal-v0",
]

ALL_PNP_REAL_NAMES = [
    "RX200PnPReal-v0",
    "RX200PnPGoalReal-v0",
    # Ned2 PnP real.
    "NED2PnPReal-v0",
    "NED2PnPGoalReal-v0",
]


# ---------------------------- Simulation Environments  ----------------------------

# ============================ RX200 Reacher Multiros Environments ============================

# Kinect-based RX200 reach — per-link FK safety + close-race guard wired in.
register(
    id="RX200ReacherSim-v0",
    entry_point="rl_environments.rx200.sim.task_envs.reach.rx200_kinect_reach_sim:RX200ReacherEnv",
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

register(
    id="RX200PushSim-v0",
    entry_point="rl_environments.rx200.sim.task_envs.push.rx200_kinect_push_sim:RX200PushEnv",
    max_episode_steps=100,
)

# Kinect-based RX200 push, goal-conditioned (Dict obs space; HER-ready).
register(
    id="RX200PushGoalSim-v0",
    entry_point="rl_environments.rx200.sim.task_envs.push.rx200_kinect_push_goal_sim:RX200PushGoalEnv",
    max_episode_steps=100,
)

# ZED2-based RX200 push (different vision sensor than kinect).
register(
    id="RX200Zed2PushSim-v0",
    entry_point="rl_environments.rx200.sim.task_envs.push.rx200_zed2_push_sim:RX200PushEnv",
    max_episode_steps=100,
)
register(
    id="RX200Zed2PushGoalSim-v0",
    entry_point="rl_environments.rx200.sim.task_envs.push.rx200_zed2_push_goal_sim:RX200PushGoalEnv",
    max_episode_steps=100,
)


# ============================ RX200 PnP Multiros Environments ============================

# 5 arm joints + 1 gripper scalar = 6 DOF action (4 DOF in EE-action mode).
# Gripper scalar drives left_finger directly; right_finger = -gripper_cmd
# inside execute_action so the fingers move symmetrically. Goal box allows
# elevated targets (z up to 0.25 m) — push restricts to ~table-top.
register(
    id="RX200PnPSim-v0",
    entry_point="rl_environments.rx200.sim.task_envs.pnp.rx200_pnp_sim:RX200PnPEnv",
    max_episode_steps=100,
)
register(
    id="RX200PnPGoalSim-v0",
    entry_point="rl_environments.rx200.sim.task_envs.pnp.rx200_pnp_goal_sim:RX200PnPGoalEnv",
    max_episode_steps=100,
)

# ZED2-based RX200 PnP (different vision sensor than kinect).
register(
    id="RX200Zed2PnPSim-v0",
    entry_point="rl_environments.rx200.sim.task_envs.pnp.rx200_zed2_pnp_sim:RX200PnPEnv",
    max_episode_steps=100,
)
register(
    id="RX200Zed2PnPGoalSim-v0",
    entry_point="rl_environments.rx200.sim.task_envs.pnp.rx200_zed2_pnp_goal_sim:RX200PnPGoalEnv",
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


# ============================ Ned2 Push Multiros Environments ============================

register(
    id="NED2PushSim-v0",
    entry_point="rl_environments.ned2.sim.task_envs.push.ned2_push_sim:NED2PushEnv",
    max_episode_steps=100,
)
register(
    id="NED2PushGoalSim-v0",
    entry_point="rl_environments.ned2.sim.task_envs.push.ned2_push_goal_sim:NED2PushGoalEnv",
    max_episode_steps=100,
)


# ============================ Ned2 PnP Multiros Environments ============================

register(
    id="NED2PnPSim-v0",
    entry_point="rl_environments.ned2.sim.task_envs.pnp.ned2_pnp_sim:NED2PnPEnv",
    max_episode_steps=100,
)
register(
    id="NED2PnPGoalSim-v0",
    entry_point="rl_environments.ned2.sim.task_envs.pnp.ned2_pnp_goal_sim:NED2PnPGoalEnv",
    max_episode_steps=100,
)


# ============================ ViperX-300 S Reacher Multiros Environments ============================

# vx300s reach (sim). Kinect-only variant for now; the Interbotix launch
# infrastructure is shared with RX200 so the env's __init__ launches
# xsarm_moveit_interface with robot_model:=vx300s dof:=6. Same per-link
# FK safety contract as RX200 / NED2 — see VX300SRobotEnv.
register(
    id="VX300SReacherSim-v0",
    entry_point="rl_environments.vx300s.sim.task_envs.reach.vx300s_reach_sim:VX300SReacherEnv",
    max_episode_steps=100,
)
register(
    id="VX300SReacherGoalSim-v0",
    entry_point="rl_environments.vx300s.sim.task_envs.reach.vx300s_reach_goal_sim:VX300SReacherGoalEnv",
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


# ============================ RX200 Push RealROS Environments ============================

# Real push relies on an externally-published cube pose topic (default
# /cube_pose, geometry_msgs/PoseStamped). When no message has been
# received within cube_pose_timeout_s, the env falls back to the YAML
# cube_init_pos and emits a throttled warning. Wire up any vision
# pipeline (aruco_ros, deep detector, mocap...) that publishes there.
register(
    id="RX200PushReal-v0",
    entry_point="rl_environments.rx200.real.task_envs.push.rx200_push_real:RX200PushEnv",
    max_episode_steps=100,
)
register(
    id="RX200PushGoalReal-v0",
    entry_point="rl_environments.rx200.real.task_envs.push.rx200_push_goal_real:RX200PushGoalEnv",
    max_episode_steps=100,
)


# ============================ RX200 PnP RealROS Environments ============================

register(
    id="RX200PnPReal-v0",
    entry_point="rl_environments.rx200.real.task_envs.pnp.rx200_pnp_real:RX200PnPEnv",
    max_episode_steps=100,
)
register(
    id="RX200PnPGoalReal-v0",
    entry_point="rl_environments.rx200.real.task_envs.pnp.rx200_pnp_goal_real:RX200PnPGoalEnv",
    max_episode_steps=100,
)


# ============================ Ned2 Reacher RealROS Environments ============================

# Per-link FK safety + joint-state staleness gate wired in; mirrors the
# RX200 reach real envs. Push + PnP real for NED2 are now implemented
# (registrations below) — pending live hardware verification.
register(
    id="NED2ReacherReal-v0",
    entry_point="rl_environments.ned2.real.task_envs.reach.ned2_reach_real:NED2ReacherEnv",
    max_episode_steps=100,
)
register(
    id="NED2ReacherGoalReal-v0",
    entry_point="rl_environments.ned2.real.task_envs.reach.ned2_reach_goal_real:NED2ReacherGoalEnv",
    max_episode_steps=100,
)


# ============================ Ned2 Push RealROS Environments ============================

# Real push: cube pose comes from an externally-published /cube_pose
# topic (geometry_msgs/PoseStamped). Fall back to YAML cube_init_pos +
# throttled warning when stale. Opt-in: pass auto_launch_cube_tracker=True
# (or --cube-tracker auto on the CLI) for one-shot rl_envs_cube_tracker
# launch under the managed-process registry.
register(
    id="NED2PushReal-v0",
    entry_point="rl_environments.ned2.real.task_envs.push.ned2_push_real:NED2PushEnv",
    max_episode_steps=100,
)
register(
    id="NED2PushGoalReal-v0",
    entry_point="rl_environments.ned2.real.task_envs.push.ned2_push_goal_real:NED2PushGoalEnv",
    max_episode_steps=100,
)


# ============================ Ned2 PnP RealROS Environments ============================

# Real PnP: same cube tracking contract as push. Gripper command is
# binary "open"/"close" via niryo_robot_tools_commander (the scalar
# action is midpoint-discretized inside the env). The mors joints are
# prismatic in metres — see ned2_pnp_task_config.yaml for the
# grasp_finger_thresh (in metres) used to derive is_grasped.
register(
    id="NED2PnPReal-v0",
    entry_point="rl_environments.ned2.real.task_envs.pnp.ned2_pnp_real:NED2PnPEnv",
    max_episode_steps=100,
)
register(
    id="NED2PnPGoalReal-v0",
    entry_point="rl_environments.ned2.real.task_envs.pnp.ned2_pnp_goal_real:NED2PnPGoalEnv",
    max_episode_steps=100,
)


# ============================ ViperX-300 S Reacher RealROS Environments ============================

# vx300s reach (real). Mirror of the RX200 reach real contract — bare
# vx300s URDF link names (`vx300s/base_link`, `vx300s/ee_gripper_link`),
# `/vx300s/...` rosparam namespace, interbotix bringup expected externally
# (xsarm_control + driver), same safety FK + joint-state staleness gate.
register(
    id="VX300SReacherReal-v0",
    entry_point="rl_environments.vx300s.real.task_envs.reach.vx300s_reach_real:VX300SReacherEnv",
    max_episode_steps=100,
)
register(
    id="VX300SReacherGoalReal-v0",
    entry_point="rl_environments.vx300s.real.task_envs.reach.vx300s_reach_goal_real:VX300SReacherGoalEnv",
    max_episode_steps=100,
)
