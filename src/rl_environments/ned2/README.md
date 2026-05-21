# NED2 environments

Reach / Push / PnP envs (sim + real, std + goal) for the Niryo Ned2 6-DOF
arm. Ported from the RX200 chain but with NED2-specific adjustments
verified against the upstream URDFs and `niryo_robot_*` packages.

This README lists what's NED2-specific (so future ports don't blindly
copy RX200 conventions) and what's still **TODO** (placeholder values
to tune against your physical workspace before first hardware run).

## Sim bring-up package

Phase A landed a sibling package, **`niryo_ned2_description_extras`**,
that mounts the Ned2 on the same desk model the RX200 sim uses + adds
a head-mount Kinect v2 + brings up `ros_control` so
`niryo_robot_follow_joint_trajectory_controller` works in Gazebo.

```bash
# Verify standalone (no RL env):
roslaunch niryo_ned2_description_extras ned2_gazebo.launch                # reach / push
roslaunch niryo_ned2_description_extras ned2_gazebo.launch gripper:=true  # push / pnp
```

The RL env's `__init__` brings up MoveIt itself; the description launch
above is just for visual sanity / standalone controller testing.

## Layout

```
ned2/
├── sim/
│   ├── robot_envs/        # NED2RobotEnv / NED2RobotGoalEnv (Gazebo)
│   └── task_envs/
│       ├── reach/         # NED2ReacherEnv / NED2ReacherGoalEnv
│       ├── push/          # NED2PushEnv   / NED2PushGoalEnv
│       └── pnp/           # NED2PnPEnv    / NED2PnPGoalEnv
└── real/
    ├── robot_envs/        # NED2RobotEnv (real) / NED2RobotGoalEnv (real)
    └── task_envs/
        ├── reach/         # NED2ReacherEnv (real) / NED2ReacherGoalEnv (real)
        ├── push/          # NED2PushEnv (real)    / NED2PushGoalEnv (real)
        └── pnp/           # NED2PnPEnv (real)     / NED2PnPGoalEnv (real)
```

12 task env files in total, 6 ids per side (sim + real) × (std + goal).
Registered in `rl_environments/__init__.py`.

## NED2 specifics (not RX200 — don't assume)

| Topic | NED2 | RX200 |
|---|---|---|
| Arm joint count | **6** (`joint_1..joint_6`) | 5 |
| Arm joint limits | from URDF — see below | wider |
| Gripper joints | **2 prismatic** (`joint_base_to_mors_{1,2}`, ±10 mm stroke) | 2 prismatic fingers (different stroke) |
| Gripper command API | **binary** `"open"`/`"close"` via `niryo_robot_tools_commander` action server | continuous position control (`set_gripper_joints([left, right])`) |
| PnP joint-mode action dim | **6 + 1 = 7** (arm + gripper scalar) | 5 + 1 = 6 |
| Arm controller topic | `/niryo_robot_follow_joint_trajectory_controller/command` | `/arm_controller/command` |
| Reference frame | `base_link` (real) / `ned2/base_link` (sim) | `rx200/base_link` |
| End-effector link | `wrist_link` (real) / `ned2/tool_link` (sim w/ gripper) | `rx200/ee_arm_link` etc. |
| Camera parent class kwarg | `use_kinect`/`use_zed2` (real), `use_camera` (sim) | `use_kinect`/`use_zed2` |

### Arm joint URDF limits (radians, from `niryo_ned2_param.urdf.xacro`)

| Joint | URDF lower | URDF upper |
|---|---|---|
| joint_1 (shoulder rotation) | -3.000 | +3.000 |
| joint_2 (arm rotation) | **-1.833** | +0.610 |
| joint_3 (elbow rotation) | -1.340 | +1.570 |
| joint_4 (forearm rotation) | -2.090 | +2.090 |
| joint_5 (wrist rotation) | -1.920 | +1.923 |
| joint_6 (hand rotation) | -2.530 | +2.530 |

The configs use slightly tighter bounds on joint_1 (`±2.949`) as a safety
margin. **joint_2 was previously set to -2.09 across all NED2 configs,
past the URDF limit — fixed in `5e6271a`.**

MoveIt allows max velocities `1.57 / 1.047 / 1.57 / 3.14 / 3.14 / 3.14`
rad/s; the configs use ~50% as a conservative cap.

### Gripper mors joints (PRISMATIC, METRES)

From `niryo_ned2_gripper1_n_camera.urdf.xacro`:

```xml
<joint name="joint_base_to_mors_1" type="prismatic">
    <limit lower="-0.01" upper="0.01" effort="1" velocity="5"/>
</joint>
```

- Total stroke: **±10 mm linear** (NOT angular).
- Both mors symmetric — total finger opening ≈ 20 mm.
- These are STATE-readable joints. They are NOT directly position-controllable.
- Commands go through `niryo_robot_tools_commander` action server which
  accepts only `OPEN_GRIPPER` / `CLOSE_GRIPPER` (tool id 11 for gripper1).

`grasp_finger_thresh` in the PnP YAML is therefore **in metres**, not
radians.

## Cube tracking on real

Same contract as RX200 real:
- Push / PnP envs subscribe to `/cube_pose` (`geometry_msgs/PoseStamped`).
- Fall back to YAML `cube_init_pos` after `cube_pose_timeout_s` of silence.
- Wire up `rl_envs_cube_tracker` (AprilTag) or any equivalent publisher.
- Optional opt-in via `auto_launch_cube_tracker=True` (or CLI
  `--cube-tracker auto`) — env roslaunches the tracker under the
  managed-process registry.

## Prerequisites (real)

- **niryo_robot_bringup** running (brings up the niryo driver,
  `niryo_robot_follow_joint_trajectory_controller`, MoveIt, and the
  tools_commander action server).
- Gripper URDF + controllers are loaded by `niryo_robot_bringup` —
  the env does NOT take a `gripper=True` kwarg on real (sim does).
- For cube tracking: any publisher emitting `geometry_msgs/PoseStamped`
  on `/cube_pose`. The shipped option is `rl_envs_cube_tracker` (AprilTag).
- Camera driver if using `use_kinect=True` or `use_zed2=True`
  (`iai_kinect2`'s `kinect2_bridge` or the ZED ROS wrapper).

## Prerequisites (sim)

- Gazebo with Niryo's `niryo_robot_gazebo` (already a transitive dep).
- A world that DOES NOT pre-spawn a `red_cube` (the env spawns it on
  reset, so a pre-baked cube triggers a name collision on the first
  `spawn_cube_in_gazebo` call).
- (Optional) A `head_mount_kinect2` model in the world when
  `use_kinect=True` — see TODO §**Camera-mount mismatch** below.

## Known TODOs

### 🟡 Missing / unverified assets

1. ~~**`ned2_workspace_only.world` is referenced but does not exist.**~~
   **Resolved (Phase A)** — `niryo_ned2_description_extras` provides
   `ned2_gazebo.launch` which mounts the Ned2 on the RX200 desk model
   (no separate world file needed). Train script docstrings updated to
   point at the new launch.

2. **No NED2-specific extrinsic YAMLs in `rl_envs_cube_tracker`.** Both
   `config/extrinsics/kinect2_to_rx200.yaml` and `zed2_to_rx200.yaml`
   reference `rx200/base_link`. Add `kinect2_to_ned2.yaml` +
   `zed2_to_ned2.yaml` (with `parent_frame: base_link`) + a
   `d405_to_ned2.yaml` + `d405_to_rx200.yaml` (and a `d405.launch` in
   the cube tracker package) before using `--cube-tracker-target-frame
   base_link` on NED2. Queued for Phase C.

### 🟡 Production placeholder values to tune

3. **`pnp_goal` static fallback** — hard-coded `[0.250, 0.000, 0.015]`
   in 4 files (push + pnp real, std + goal). Carried from the RX200
   template; tune for your NED2 workspace_1 pad position. Search for
   `# TODO: confirm NED2 {push,pnp} static goal pose`.

4. **Cube spawn position mismatch** — `cube_init_vector = np.array([0.180,
   0.000, 0.015])` hard-coded in NED2 sim push (both std + goal). YAMLs
   say `cube_init_pos: [0.20, 0.0, 0.015]`. The hard-coded value is the
   fallback; tune both consistently.

5. **`grasp_finger_thresh = 0.0` (metres)** in `ned2_pnp_task_config.yaml`.
   This is the position-ceiling for treating `joint_base_to_mors_1` as
   "closed on something". The exact value depends on the encoder
   reading when fingers actually grip a cube vs empty — empirically
   calibrate on hardware before relying on `is_grasped`.

6. **`init_pos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`** is the URDF-zero pose
   (arm fully extended forward and up). Niryo's conventional "sleep"
   pose folds the arm down toward the base. Consider switching to
   something like `[0.0, 0.55, -1.5, 0.0, 0.0, 0.0]` (radians) if your
   workspace doesn't have clearance for the extended pose.

### 🟡 Pre-existing inconsistencies worth a second look

7. **`ned2_reach_goal_sim.py` default `action_cycle_time = 0.800`** vs
   `0.100` in every other NED2 sim task env. Pre-existing — possibly
   intentional for slower reach demos, possibly a typo.

8. **`max_joint_vel[5] = 1.775` rad/s** (joint_6) across all NED2 configs.
   MoveIt allows 3.14 rad/s, so this is ~57% — slightly above the
   ~50% safety margin the other joints follow (`0.785 / 0.5235 / 0.785
   / 1.57 / 1.57 / 1.775`). Either intentional or a copy-paste artefact;
   confirm against Niryo's published velocity spec.

9. ~~**Camera-mount mismatch (sim)**~~ **Resolved (Phase A)** —
   `niryo_ned2_description_extras/urdf/ned2_kinect.urdf.xacro` mounts a
   head_mount_kinect2 at the same pose RX200 uses, so the RL env's
   subscribers to `/head_mount_kinect2/{rgb,depth}/image_raw` now have
   a real publisher in sim. The Ned2's built-in wrist camera is also
   enabled (publishes on `/gazebo_camera/*`) — unused by the cube
   tracker but available for future EE-mounted perception.

10. **`/head_mount_kinect2/...` on real** — same topic name on real;
    expects `iai_kinect2/kinect2_bridge` running with `base_name:=head_mount_kinect2`
    (NOT the default `kinect2`). Either rename your bridge instance or
    retarget the subscribers.

11. **`ned2_reach_real` timing defaults differ from push/pnp real.**
    Reach: `environment_loop_rate=None`, `action_cycle_time=0.0` (no
    timer-driven loop by default). Push/PnP: `=10`, `=0.100`. Matches
    the RX200 reach real pattern but means reach real users have to
    pass `environment_loop_rate=10.0` explicitly to get the same
    timer-driven safety hooks (FK + staleness gate) running.

12. **MoveIt `reference_frame: world`** in
    `niryo_robot_arm_commander/config/default.yaml`, but NED2 robot
    env uses `ref_frame = "base_link"`. URDF likely anchors `base_link`
    coincident with `world` at origin (no offset), but worth a sanity
    check if you redeploy NED2 in a multi-robot world where the base
    isn't at origin.

### ✅ Already fixed

Earlier session:
- `ned2_reach_goal_sim.py` smoothing vector size 5 → 6.
- "5 elements" docstrings in all 4 robot env files → "6 elements".
- `joint_2` min URDF violation in 3 configs (-2.09 → -1.833).
- mors joint type/units corrected (revolute/radians → prismatic/metres)
  in PnP config + 4 comments across 2 task env files.

Phase A:
- New `niryo_ned2_description_extras` package with Gazebo launch
  (mounts Ned2 on RX200 desk, head-mount Kinect v2, ros_control with
  Niryo PID gains, optional gripper variant).
- SRDF/URDF name mismatch resolved (`niryo_ned2`).

## Maintenance conventions for NED2

Whenever you touch a NED2 env file, check:

1. **Joint count is 6**, not 5. Search for `np.zeros(5)`, `len(jv) < 5`,
   `5 elements`, `range(5)` — those are RX200-isms.
2. **Mors comments say PRISMATIC and METRES**, not revolute or radians.
3. **Action dim** is 6 (reach/push joint-mode), 7 (PnP joint-mode + gripper),
   or 3 / 4 (EE-mode + optional gripper).
4. **Real envs use bare URDF link names** (`base_link`, `wrist_link`) — no
   `ned2/` prefix. Sim envs use the `ned2/` prefix.
5. **`use_kinect`/`use_zed2` on real, `use_camera` on sim parent** — only
   the sim parent was renamed (commit `5e6271a` updated the strict-safety
   flag, not the camera kwarg).
6. **`/ned2/...` rosparam namespace** — never `/rx200/...`.
7. **`move_NED2_object`** for the MoveIt handle.
8. **`gripper=True` to super on sim only** — real env's super does NOT
   take this kwarg; the gripper bringup is external via
   `niryo_robot_bringup`.

## Contact

[j.kapukotuwa@research.ait.ie](mailto:j.kapukotuwa@research.ait.ie)
