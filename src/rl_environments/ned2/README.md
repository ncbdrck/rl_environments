# NED2 environments — working notes

Reach / Push / PnP envs (sim + real, std + goal) for the Niryo Ned2
6-DOF arm. This README is the *internal* punch list — open TODOs,
placeholder values to tune, and real-world checks to run before
shipping. Public-facing docs live in `rl_environments/README.md`.

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

12 task env files in total, 12 ids (6 ids per side × std + goal).
Registered in `rl_environments/__init__.py`.

## NED2 specifics (not RX200 — don't assume)

| Topic | NED2 | RX200 |
|---|---|---|
| Arm joint count | **6** (`joint_1..joint_6`) | 5 |
| Gripper joints | **2 prismatic** (`joint_base_to_mors_{1,2}`, ±10 mm stroke) | 2 prismatic fingers (different stroke) |
| Gripper command API | **binary** `"open"`/`"close"` (real: `niryo_robot_tools_commander`; sim: direct `/gazebo_tool_commander/follow_joint_trajectory` publish) | continuous position control |
| PnP joint-mode action dim | **6 + 1 = 7** | 5 + 1 = 6 |
| Arm controller topic | `/niryo_robot_follow_joint_trajectory_controller/command` | `/arm_controller/command` |
| Reference frame | `base_link` (URDF links are bare on both sim + real) | `rx200/base_link` |
| End-effector link | `tool_link` (FK target + MoveIt's planning EE; base→tool chain has 6 movable joints matching the 6-DOF action) | `rx200/ee_arm_link` etc. |
| Gazebo link-state lookup | `ned2/<link>` (model-qualified form for `get_model_state`) | `rx200/<link>` |

### Arm joint URDF limits (radians, from `niryo_ned2_param.urdf.xacro`)

| Joint | URDF lower | URDF upper |
|---|---|---|
| joint_1 (shoulder rotation) | -3.000 | +3.000 |
| joint_2 (arm rotation) | -1.833 | +0.610 |
| joint_3 (elbow rotation) | -1.340 | +1.570 |
| joint_4 (forearm rotation) | -2.090 | +2.090 |
| joint_5 (wrist rotation) | -1.920 | +1.923 |
| joint_6 (hand rotation) | -2.530 | +2.530 |

Configs use slightly tighter bounds on joint_1 (`±2.949`) as a safety
margin. MoveIt allows max velocities `1.57 / 1.047 / 1.57 / 3.14 / 3.14
/ 3.14` rad/s; configs use ~50% as a conservative cap.

### Gripper mors joints (PRISMATIC, METRES)

```xml
<joint name="joint_base_to_mors_1" type="prismatic">
    <limit lower="-0.01" upper="0.01" effort="1" velocity="5"/>
</joint>
```

- Total stroke: **±10 mm linear** (NOT angular).
- Both mors symmetric — total finger opening ≈ 20 mm.
- STATE-readable but NOT directly position-controllable on real (only
  via the `niryo_robot_tools_commander` binary OPEN/CLOSE action).
- `grasp_finger_thresh` in the PnP YAML is therefore **in metres**.

### Wrist camera (opt-in)

Off by default. Pass `use_wrist_camera=True` (kwarg or `--wrist-camera`)
to subscribe. Decoded frame is exposed as `self.cv_image_wrist`.

| Side | Topic | Type |
|---|---|---|
| Sim | `/gazebo_camera/image_raw` | `sensor_msgs/Image` |
| Real | `/niryo_robot_vision/compressed_video_stream` | `sensor_msgs/CompressedImage` (needs `niryo_robot_vision` running) |

## Open TODOs — placeholder values to tune before first hardware run

1. **`pnp_goal` static fallback** — hard-coded `[0.250, 0.000, 0.015]`
   in 4 files (push + pnp real, std + goal). Carried from the RX200
   template; tune for the NED2 workspace_1 pad position. Search for
   `# TODO: confirm NED2 {push,pnp} static goal pose`.

2. **Cube spawn position (sim push) — hard-coded vs YAML** —
   `cube_init_vector = np.array([0.180, 0.000, 0.015])` hard-coded in
   NED2 sim push (both std + goal). The matching sim YAMLs use
   `cube_init_pos: [0.25, 0.0, 0.015]`. Real-side fallback already
   aligned to YAML; tune the sim hard-coded value if you want them to
   match.

3. **`grasp_finger_thresh = 0.0` (metres)** in `ned2_pnp_task_config.yaml`.
   Position-ceiling for treating `joint_base_to_mors_1` as "closed on
   something". Calibrate empirically on hardware before relying on
   `is_grasped`.

4. **`init_pos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`** is the URDF-zero pose
   (arm fully extended forward and up). Niryo's conventional "sleep"
   pose folds the arm down. Consider switching to something like
   `[0.0, 0.55, -1.5, 0.0, 0.0, 0.0]` if your workspace lacks clearance
   for the extended pose.

## Open TODOs — pre-existing inconsistencies worth a second look

5. **`ned2_reach_goal_sim.py` default `action_cycle_time = 0.800`** vs
   `0.100` in every other NED2 sim task env. Possibly intentional for
   slower reach demos; possibly a typo.

6. **`max_joint_vel[5] = 1.775` rad/s** (joint_6) across all NED2 configs.
   MoveIt allows 3.14 rad/s, so this is ~57% — slightly above the
   ~50% margin the other joints follow. Confirm against Niryo's
   published velocity spec or align to ~50%.

7. **MoveIt `reference_frame: world`** in
   `niryo_robot_arm_commander/config/default.yaml`, but the NED2 robot
   env uses `ref_frame = "base_link"`. URDF likely anchors `base_link`
   coincident with `world` at origin, but worth a sanity check if you
   redeploy NED2 in a multi-robot world where the base isn't at origin.

## Real-world checks (run before / during first hardware bring-up)

8. **`/head_mount_kinect2/...` on real** — same topic name on real;
   expects `iai_kinect2/kinect2_bridge` running with
   `base_name:=head_mount_kinect2` (NOT the default `kinect2`).
   Either rename your bridge instance or retarget the subscribers.

9. **`ned2_reach_real` timing defaults differ from push/pnp real.**
   Reach: `environment_loop_rate=None`, `action_cycle_time=0.0`. Push/PnP:
   `=10`, `=0.100`. Matches the RX200 reach real pattern but means
   reach-real users must pass `environment_loop_rate=10.0` explicitly
   to get the timer-driven safety hooks (FK + staleness gate) running.

10. **Extrinsic YAMLs are placeholders.** The 6 files under
    `rl_envs_cube_tracker/config/extrinsics/` ({kinect2,zed2,d405}_to_{rx200,ned2})
    use placeholder XYZ + RPY values, **not** real measurements.
    Calibrate against the physical camera mount before relying on
    `--cube-tracker-target-frame base_link` for NED2 or
    `rx200/base_link` for RX200.

11. **End-to-end Gazebo verification still pending.** Run e.g.:

    ```bash
    roslaunch niryo_ned2_description_extras ned2_gazebo.launch gripper:=true
    rosrun rl_training_validation ned2_pnp_train_sim.py --gazebo-gui
    ```

    Confirm: ros_control loads, MoveIt accepts trajectories, no PID
    sag, gripper open/close works through the direct
    `/gazebo_tool_commander/...` publish path, cube spawn succeeds.

12. **Grasp stability in Gazebo.** Pure Gazebo grasping a 0.02 m cube
    with the Niryo's ~0.02 m gripper opening is marginal. For reliable
    sim training, install JenniferBuehler's
    [`gazebo_grasp_fix`](https://github.com/JenniferBuehler/gazebo-pkgs)
    plugin and add it to the Niryo URDF, OR tune cube + finger-tip
    friction toward near-stiction.

## Maintenance conventions for NED2

Whenever you touch a NED2 env file, check:

1. **Joint count is 6**, not 5. Search for `np.zeros(5)`, `len(jv) < 5`,
   `5 elements`, `range(5)` — those are RX200-isms.
2. **Mors comments say PRISMATIC and METRES**, not revolute or radians.
3. **Action dim** is 6 (reach/push joint-mode), 7 (PnP joint-mode + gripper),
   or 3 / 4 (EE-mode + optional gripper).
4. **Real envs use bare URDF link names** (`base_link`, `wrist_link`) — no
   `ned2/` prefix. Sim envs use the `ned2/` prefix.
5. **`/ned2/...` rosparam namespace** — never `/rx200/...`.
6. **`move_NED2_object`** for the MoveIt handle.
7. **`gripper=True` to super on sim only** — real env's super does NOT
   take this kwarg; the gripper bringup is external via `niryo_robot_bringup`.
