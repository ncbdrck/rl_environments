# NED2 environments

Reach / Push / PnP envs (sim + real, std + goal) for the Niryo Ned2
6-DOF arm.

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

12 ids in total (6 tasks × std + goal). Registered in
`rl_environments/__init__.py`.

## NED2 specifics

| Topic | NED2 |
|---|---|
| Arm joint count | 6 (`joint_1..joint_6`) |
| Gripper joints | 2 prismatic (`joint_base_to_mors_{1,2}`, ±10 mm stroke) |
| Gripper command API | binary `"open"`/`"close"` (real: `niryo_robot_tools_commander`; sim: direct `/gazebo_tool_commander/follow_joint_trajectory` publish) |
| PnP joint-mode action dim | 6 + 1 = 7 |
| Arm controller topic | `/niryo_robot_follow_joint_trajectory_controller/command` |
| Reference frame | `base_link` (URDF links are bare on both sim + real) |
| End-effector link | `tool_link` (FK target + MoveIt planning EE) |
| Gazebo link-state lookup | `ned2/<link>` (model-qualified form for `get_model_state`) |

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

### Gripper mors joints (prismatic, metres)

```xml
<joint name="joint_base_to_mors_1" type="prismatic">
    <limit lower="-0.01" upper="0.01" effort="1" velocity="5"/>
</joint>
```

- Total stroke: ±10 mm linear (not angular).
- Both mors symmetric — total finger opening ≈ 20 mm.
- State-readable but not directly position-controllable on real (only
  via the `niryo_robot_tools_commander` binary OPEN/CLOSE action).
- `grasp_finger_thresh` in the PnP YAML is therefore in metres.

### Wrist camera (opt-in)

Off by default. Pass `use_wrist_camera=True` (kwarg or `--wrist-camera`)
to subscribe. Decoded frame is exposed as `self.cv_image_wrist`.

| Side | Topic | Type |
|---|---|---|
| Sim | `/gazebo_camera/image_raw` | `sensor_msgs/Image` |
| Real | `/niryo_robot_vision/compressed_video_stream` | `sensor_msgs/CompressedImage` (needs `niryo_robot_vision` running) |

## Conventions when editing NED2 files

- Joint count is 6 (don't blindly port 5-element arrays from RX200).
- Mors joints are prismatic, in metres (not revolute or radians).
- Action dim is 6 (reach/push joint-mode), 7 (PnP joint-mode + gripper),
  or 3 / 4 (EE-mode + optional gripper).
- Real envs use bare URDF link names (`base_link`, `wrist_link`); sim
  envs use the `ned2/` prefix.
- `/ned2/...` rosparam namespace.
- `move_NED2_object` for the MoveIt handle.
- `gripper=True` to super on sim only — real env's super does not take
  this kwarg; the gripper bring-up is external via `niryo_robot_bringup`.
