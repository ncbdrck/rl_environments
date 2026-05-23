# UR5e environments

Reach / Push / PnP envs (sim + real, std + goal) for the Universal
Robots UR5e + Robotiq 2F-85 gripper.

## Layout

```
ur5e/
├── sim/
│   ├── robot_envs/        # UR5eRobotEnv / UR5eRobotGoalEnv (Gazebo)
│   └── task_envs/
│       ├── reach/         # UR5eReacherEnv / UR5eReacherGoalEnv
│       ├── push/          # UR5ePushEnv   / UR5ePushGoalEnv
│       └── pnp/           # UR5ePnPEnv    / UR5ePnPGoalEnv
└── real/
    ├── robot_envs/        # UR5eRobotEnv (real) / UR5eRobotGoalEnv (real)
    └── task_envs/
        ├── reach/         # UR5eReacherEnv (real) / UR5eReacherGoalEnv (real)
        ├── push/          # UR5ePushEnv (real)    / UR5ePushGoalEnv (real)
        └── pnp/           # UR5ePnPEnv (real)     / UR5ePnPGoalEnv (real)
```

12 ids in total (6 tasks × std + goal). Registered in
`rl_environments/__init__.py`.

## UR5e specifics

| Topic | UR5e |
|---|---|
| Arm joint count | 6 (`shoulder_pan_joint`, `shoulder_lift_joint`, `elbow_joint`, `wrist_1_joint`, `wrist_2_joint`, `wrist_3_joint`) |
| Gripper joints | 1 actuated (`robotiq_85_left_knuckle_joint`); the other 5 finger joints follow via URDF mimic linkage |
| Gripper command API | continuous position on the knuckle in [gripper_min ≈ 0.0, gripper_max ≈ 0.8] |
| PnP joint-mode action dim | 6 + 1 = 7 (arm + gripper scalar) |
| Arm controller topic | `/ur5e/arm_controller/command` (sim + real, after the `/ur5e/` namespace wrapper for the upstream MoveIt config) |
| Reference frame | `base_link` (bare URDF link names; no `ur5e/` prefix unlike the Interbotix robots) |
| End-effector link | `ee_link` (MoveIt SRDF group `arm` ends here) |
| Gazebo link-state lookup | `ur5e/<link>` (model-qualified form for `get_model_state`) |
| Gripper "open" / "closed" | OPEN = gripper_min, CLOSED = gripper_max — INVERTED from VX300S where OPEN = gripper_max |
| `grasp_finger_thresh` semantics | LOWER bound on the knuckle ("closed enough" when `knuckle_pos > grasp_finger_thresh`), opposite of VX300S's `<` |

Sim brings up the description-extras package
([`ur5e_description_extras`](https://github.com/ncbdrck/ur5e_description_extras)),
which mounts the arm on a 4-legged `ur5_base` (~0.59 m) next to a
`cafe_table` at world (0.7, 0, 0) with a head-mount Kinect v2. Real
assumes a `ur5e_real.launch` wrapper (TBD at the lab) that brings up
`ur_robot_driver` + a Robotiq driver + MoveIt under `/ur5e`.

## Initial pose

Every UR5e task env resets to a folded-upright pose:

```python
self.init_pos = np.array([0.0, -1.5707, 1.5707, -1.5707, -1.5707, 0.0],
                         dtype=np.float32)
```

UR5e's all-zeros URDF pose puts the arm horizontal at base height
(z = 0.59), inside the cafe-table column — MoveIt's plan-from-collision
fails before it tries, so we bypass it: sim snaps joints via Gazebo's
`SetModelConfiguration` (pause → snap → unpause), then drives both
arm and gripper through the low-level trajectory controllers; real
publishes directly to the controllers since the physical arm starts
wherever it was left, not at URDF zero.

## Safety geometry

UR5e mounts at z = 0.59 on the ur5_base, with the workspace cafe-table
SEPARATE at world (0.7, 0, 0) and the table top at z = 0.775. Unlike
RX200 / NED2 / VX300S (flush-mounted on the table), the safety check
enforces:

1. A base floor for every link — `z_world ≥ base_z + safety_z_margin`.
2. A table floor only when the link's XY lands inside the cafe-table
   footprint — `z_world ≥ table_top_z + safety_z_margin`.

Both `base_z`, `table_top_z`, and the table footprint are rosparams
(see `config/ur5e_*_task_config.yaml`).

## Conventions when editing UR5e files

- Joint count is 6 (don't blindly port 5-element arrays from RX200).
- Robotiq 2F-85 has ONE actuated knuckle, not a prismatic finger pair.
  - `move_gripper_joints` publishes a single-element trajectory, NOT
    `[gripper_cmd, -gripper_cmd]`.
  - `is_grasped` looks up `robotiq_85_left_knuckle_joint` in
    `joint_state.name`, NOT `left_finger`.
- Grasp comparison is `knuckle_pos > grasp_finger_thresh` (inverted
  from VX300S's `<`).
- Use `gripper_open_value` / `gripper_closed_value` rosparams when
  code needs "the open value" or "the closed value" — `gripper_min`
  and `gripper_max` are NUMERICAL bounds, not semantic names.
- `move_UR5E_object` for the MoveIt handle (note caps: UR5E, not UR5e
  — the sed pass standardised this so all references match).
- MoveIt planning groups are `arm` and `gripper` (from the upstream
  `ur5e_robotiq_85_moveit_config` SRDF), not `interbotix_arm` /
  `interbotix_gripper`.
- Bare URDF link names everywhere (`base_link`, `ee_link`,
  `shoulder_link`, `upper_arm_link`, `forearm_link`,
  `wrist_{1,2,3}_link`). The Interbotix `vx300s/` / `rx200/` prefix
  does NOT apply.
