# RX200 environments — working notes

Reach / Push / PnP envs (sim + real, std + goal) for the Trossen
ReactorX-200 5-DOF arm, with kinect and zed2 sim variants. This README
is the *internal* punch list — open TODOs, placeholder values to tune,
and real-world checks to run before / during hardware bring-up. Public
docs live in `rl_environments/README.md`.

## Layout

```
rx200/
├── sim/
│   ├── robot_envs/        # RX200RobotEnv / *_goal_sim / *_zed2 / *_goal_sim_zed2
│   └── task_envs/
│       ├── reach/         # RX200ReacherEnv / RX200ReacherGoalEnv
│       ├── push/          # RX200PushEnv   / RX200PushGoalEnv
│       └── pnp/           # RX200PnPEnv    / RX200PnPGoalEnv
└── real/
    ├── robot_envs/        # RX200RobotEnv (real) / RX200RobotGoalEnv (real)
    └── task_envs/
        ├── reach/         # RX200ReacherEnv (real) / RX200ReacherGoalEnv (real)
        ├── push/          # RX200PushEnv (real)    / RX200PushGoalEnv (real)
        └── pnp/           # RX200PnPEnv (real)     / RX200PnPGoalEnv (real)
```

18 ids in total (kinect: reach/push/pnp × std/goal × sim/real = 12;
zed2: reach/push/pnp × std/goal × sim only = 6). Registered in
`rl_environments/__init__.py`.

## RX200 specifics

| Topic | RX200 |
|---|---|
| Arm joint count | **5** (`waist`, `shoulder`, `elbow`, `wrist_angle`, `wrist_rotate`) |
| Gripper joints | **2 prismatic fingers** with continuous position control |
| PnP joint-mode action dim | **5 + 1 = 6** (arm + gripper scalar) |
| Arm controller topic | `/arm_controller/command` (sim) / interbotix driver (real) |
| Reference frame | `rx200/base_link` (sim+real) |
| Gripper command API | continuous position: `set_gripper_joints([left, right])` |
| `grasp_finger_thresh` | **metres** (left_finger position ceiling) |

Sim brings up an own kinect2 or zed2 head-mount via
`reactorx200_description`. Real assumes `iai_kinect2/kinect2_bridge`
running with `base_name:=head_mount_kinect2` (the env subscribers use
that prefix).

## Open TODOs — placeholder values to tune before first hardware run

1. **`pnp_goal` static fallback** — hard-coded `[0.25, 0.0, 0.015]` in
   `rx200_pnp_task_config.yaml`. Cube-tracker-driven path overrides
   this; the static value is only used when no tracker is publishing.
   Tune for the physical workspace_1 pad position.

2. **`grasp_finger_thresh = 0.020` m** in `rx200_pnp_task_config.yaml`.
   Position-ceiling for "closed on something". Calibrate against the
   actual encoder reading when the gripper grasps a cube on hardware.

## Real-world checks (run before / during first hardware bring-up)

3. **Extrinsic YAMLs are placeholders.** Files under
   `rl_envs_cube_tracker/config/extrinsics/` (`kinect2_to_rx200.yaml`,
   `zed2_to_rx200.yaml`, `d405_to_rx200.yaml`) use placeholder XYZ +
   RPY values, **not** real measurements. Calibrate against the
   physical camera mount before relying on
   `--cube-tracker-target-frame rx200/base_link`.

4. **End-to-end real sweep.** After interbotix driver bring-up + tracker:

    ```bash
    rosparam set /allow_real_robot_motion true
    rosrun rl_training_validation rx200_reach_train_real.py --allow-real-robot-motion
    # then push / pnp with --cube-tracker auto if the tracker isn't running
    ```

    Confirm: safety FK gate fires on out-of-table targets, staleness
    gate fires when joint_states stop, gripper open/close grips the
    cube reliably, episode reset returns to a safe pose.

5. **Cube-tracker frame match.** When the tracker publishes into
   `rx200/base_link`, the env's `cube_pose_topic` consumer must agree.
   Default `/cube_pose` works; if you retarget the topic, also confirm
   the frame the publisher emits matches `cube_tracker_target_frame`.

6. **Goal-tolerance violation warnings.** Cosmetic on sim (env publishes
   directly, bypassing MoveIt's tolerance check). If they appear on
   real, that means a trajectory was sent to the real MoveIt action
   server that didn't converge — investigate, don't dismiss.

## Maintenance conventions for RX200

Whenever you touch an RX200 env file:

1. **Joint count is 5**, not 6 — NED2 is 6 (don't blindly port between).
2. **Gripper is continuous-control**, not binary like NED2. Action dim
   for PnP is 5+1 (arm + gripper scalar), and `grasp_finger_thresh`
   reads as a position threshold in metres.
3. **All envs use `rx200/` prefix** for link names and rosparam
   namespaces (sim AND real).
4. **`move_RX200_object`** for the MoveIt handle.
5. **`use_kinect` vs `use_zed2`** — kinect is the default; zed2
   variants have their own robot_env files and registry ids
   (`RX200Zed2...Sim-v0`).
