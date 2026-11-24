```bash
uv run --extra importers isaaclab train \
  --rl_library rsl_rl \
  --task RobotLab-Isaac-Velocity-Flat-E1-Wheel-v0 \
  --num_envs 256 \
  --max_iterations 5000 \
  --viz none \
  physics=newton_mjwarp
  
uv run --extra importers isaaclab play \
  --rl_library rsl_rl \
  --task RobotLab-Isaac-Velocity-Flat-E1-Wheel-v0 \
  --checkpoint latest \
  --num_envs 4 \
  --viz newton_gl \
  physics=newton_mjwarp env.commands.base_velocity.rel_standing_envs=1.0