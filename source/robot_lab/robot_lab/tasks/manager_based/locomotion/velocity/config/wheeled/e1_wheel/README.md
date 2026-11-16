# E1 wheeled velocity tracking

These tasks use `robot_lab.assets.e1_wheel.E1_WHEEL_CFG` and the headless
`E1_DOG_wheel_no_head.urdf` asset.

| Terrain | Task ID | RSL-RL experiment |
| --- | --- | --- |
| Flat | `RobotLab-Isaac-Velocity-Flat-E1-Wheel-v0` | `e1_wheel_flat` |
| Rough | `RobotLab-Isaac-Velocity-Rough-E1-Wheel-v0` | `e1_wheel_rough` |

The previous `RobotLab-Isaac-Velocity-{Flat,Rough}-e1-wheeled` IDs remain aliases.

## Control and observations

- Actions contain 16 leg position offsets followed by four wheel velocity targets.
- Leg order is FR, FL, RR, RL; each leg uses thigh, hip roll, hip yaw, calf.
- Wheel order is FR, FL, RR, RL.
- Position scales are 0.25 rad for thigh/calf and 0.125 rad for hip roll/yaw.
- Wheel velocity scale is 10 rad/s; targets are clipped at the URDF's 13.09 rad/s.
- The RSL-RL wrapper clips normalized actions to `[-1, 1]`.
- Wheel angles are excluded from observations and joint position limit rewards.
- Forward commands range from -0.8 to 0.8 m/s; yaw commands from -0.8 to 0.8 rad/s.
  Lateral commands are zero for the initial nonholonomic velocity task.
- Resets start near the nominal upright stance. Physical parameter and gain
  randomization use moderate ranges; base contact terminates an episode.

The source URDF retains its zero/zero wheel position limits. The E1 asset spawner
overrides only the four wheel joints to unlimited rotation in the spawned USD
before physics initialization, because Newton MJWarp rejects zero-width joint
ranges. It does not modify the URDF, add collision geometry, or change leg limits.

## Run from the repository root

```bash
uv run --extra importers isaaclab list_envs --keyword E1-Wheel

uv run --extra importers isaaclab zero_agent \
  --task RobotLab-Isaac-Velocity-Flat-E1-Wheel-v0 \
  --num_envs 4 --max_steps 100 --viz newton_gl physics=newton_mjwarp

uv run --extra importers isaaclab train --rl_library rsl_rl \
  --task RobotLab-Isaac-Velocity-Flat-E1-Wheel-v0 \
  --num_envs 256 --max_iterations 5000 --viz none physics=newton_mjwarp

uv run --extra importers isaaclab play --rl_library rsl_rl \
  --task RobotLab-Isaac-Velocity-Flat-E1-Wheel-v0 \
  --checkpoint /absolute/path/to/model.pt --num_envs 4 \
  --viz newton_gl physics=newton_mjwarp
```

Use the Rough task ID to train on generated terrain. It retains terrain curriculum
and terrain scans for the critic; the Flat task removes both ray casters.
Logs are written under `logs/rsl_rl/e1_wheel_flat` or `logs/rsl_rl/e1_wheel_rough`.
CusRL configurations are optional and require the `cusrl` extra and its own launcher.

The reward weights, initial pose, and PD gains are starting configurations.
Smoke tests establish that the task runs, not policy convergence or sim-to-real quality.
