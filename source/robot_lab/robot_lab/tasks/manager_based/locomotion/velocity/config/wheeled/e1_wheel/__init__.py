# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents

for terrain in ("Flat", "Rough"):
    # Retain the original local task IDs as aliases for existing commands.
    for task_id in (
        f"RobotLab-Isaac-Velocity-{terrain}-E1-Wheel-v0",
        f"RobotLab-Isaac-Velocity-{terrain}-e1-wheeled",
    ):
        gym.register(
            id=task_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.{terrain.lower()}_env_cfg:E1Wheel{terrain}EnvCfg",
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:E1Wheel{terrain}PPORunnerCfg",
                "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:E1Wheel{terrain}TrainerCfg",
            },
        )
