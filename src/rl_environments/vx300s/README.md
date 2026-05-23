# VX300S environments

Reach / Push / PnP envs (sim + real, std + goal) for the Trossen
ViperX-300 S 6-DOF arm.

## Layout

```
vx300s/
├── sim/
│   ├── robot_envs/        # VX300SRobotEnv / VX300SRobotGoalEnv (Gazebo)
│   └── task_envs/
│       ├── reach/         # VX300SReacherEnv / VX300SReacherGoalEnv
│       ├── push/          # VX300SPushEnv   / VX300SPushGoalEnv
│       └── pnp/           # VX300SPnPEnv    / VX300SPnPGoalEnv
└── real/
    ├── robot_envs/        # VX300SRobotEnv (real) / VX300SRobotGoalEnv (real)
    └── task_envs/
        ├── reach/         # VX300SReacherEnv (real) / VX300SReacherGoalEnv (real)
        ├── push/          # VX300SPushEnv (real)    / VX300SPushGoalEnv (real)
        └── pnp/           # VX300SPnPEnv (real)     / VX300SPnPGoalEnv (real)
```

12 ids in total (6 tasks × std + goal). Registered in
`rl_environments/__init__.py`.

## VX300S specifics

| Topic | VX300S |
|---|---|
| Arm joint count | 6 (`waist`, `shoulder`, `elbow`, `forearm_roll`, `wrist_angle`, `wrist_rotate`) |
| Gripper joints | 2 prismatic fingers (`left_finger`, `right_finger`) with continuous position control |
| Gripper command API | continuous position; `right_finger = -left_finger` set just before publishing |
| PnP joint-mode action dim | 6 + 1 = 7 (arm + gripper scalar) |
| Arm controller topic | `/vx300s/arm_controller/command` (sim) / interbotix driver (real) |
| Reference frame | `vx300s/base_link` (sim+real) |
| End-effector link | `vx300s/ee_gripper_link` |
| Gripper "open" / "closed" | OPEN = gripper_max, CLOSED = gripper_min (positive left, negative right open the fingers — same convention as RX200) |
| `grasp_finger_thresh` semantics | UPPER bound on the left finger ("closed enough" when `left_finger_pos < grasp_finger_thresh`) |

Sim brings up the description-extras package
([`viperx300s_description`](https://github.com/ncbdrck/viperx300s_description)),
which mounts the arm on a cafe-table at z=0.78 with a head-mount
Kinect v2. Real expects the Interbotix `xsarm_moveit_interface`
launch with `robot_model:=vx300s dof:=6`.

## Conventions when editing VX300S files

- Joint count is 6 (UR5e is also 6 but with different names —
  shoulder_pan / shoulder_lift / wrist_1..3 vs Interbotix's
  waist / shoulder / forearm_roll / wrist_{angle,rotate}).
- Gripper has 2 prismatic finger joints commanded as
  `[left, right]` with `right = -left` (mirror).
- Grasp comparison is `left_finger_pos < grasp_finger_thresh`
  (UR5e/Robotiq inverts this to `>`).
- All envs use the `vx300s/` prefix for link names + rosparam
  namespaces (sim and real).
- `move_VX300S_object` for the MoveIt handle.
- MoveIt planning groups: `interbotix_arm` + `interbotix_gripper`.
