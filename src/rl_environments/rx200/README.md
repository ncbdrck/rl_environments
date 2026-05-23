# RX200 environments

Reach / Push / PnP envs (sim + real, std + goal) for the Trossen
ReactorX-200 5-DOF arm, with kinect and zed2 sim variants.

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
| Arm joint count | 5 (`waist`, `shoulder`, `elbow`, `wrist_angle`, `wrist_rotate`) |
| Gripper joints | 2 prismatic fingers with continuous position control |
| PnP joint-mode action dim | 5 + 1 = 6 (arm + gripper scalar) |
| Arm controller topic | `/arm_controller/command` (sim) / interbotix driver (real) |
| Reference frame | `rx200/base_link` (sim+real) |
| Gripper command API | continuous position: `set_gripper_joints([left, right])` |
| `grasp_finger_thresh` | metres (left_finger position ceiling) |

Sim brings up its own kinect2 or zed2 head-mount via
`reactorx200_description`. Real assumes `iai_kinect2/kinect2_bridge`
running with `base_name:=head_mount_kinect2`.

## Conventions when editing RX200 files

- Joint count is 5 (NED2 is 6 — don't blindly port between).
- Gripper is continuous-control, not binary like NED2. PnP action dim
  is 5+1 (arm + gripper scalar); `grasp_finger_thresh` is a position
  threshold in metres.
- All envs use the `rx200/` prefix for link names and rosparam
  namespaces (sim and real).
- `move_RX200_object` for the MoveIt handle.
- `use_kinect` vs `use_zed2` — kinect is the default; zed2 variants
  have their own robot_env files and registry ids (`RX200Zed2...Sim-v0`).
