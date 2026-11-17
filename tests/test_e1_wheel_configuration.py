# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Check that discovered E1 tasks target the authored robot instead of the Go2W template."""

import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest


@pytest.mark.parametrize("terrain", ["Flat", "Rough"])
def test_e1_wheel_configuration_matches_urdf(terrain):
    from isaaclab_tasks.utils import resolve_task_config

    import robot_lab.tasks  # noqa: F401

    task = f"RobotLab-Isaac-Velocity-{terrain}-E1-Wheel-v0"
    env, agent = resolve_task_config(task, "rsl_rl_cfg_entry_point", overrides=[])
    env.validate()
    asset_path = Path(env.scene.robot.spawn.asset_path)
    assert asset_path.name == "E1_DOG_wheel_no_head.urdf"
    robot = ET.parse(asset_path).getroot()
    joints = {joint.get("name"): joint for joint in robot.findall("joint") if joint.get("type") != "fixed"}
    leg_names = env.actions.joint_pos.joint_names
    wheel_names = env.actions.joint_vel.joint_names
    assert len(leg_names) == 16 and len(wheel_names) == 4
    assert not set(leg_names) & set(wheel_names)
    assert set(leg_names + wheel_names) == set(joints)

    # Go2W observes all joints and zeros wheel angles instead of dropping their dimensions.
    from robot_lab.tasks.manager_based.locomotion.velocity import mdp

    for group in (env.observations.policy, env.observations.critic):
        assert group.joint_pos.func is mdp.joint_pos_rel_without_wheel
        assert group.joint_pos.params["wheel_asset_cfg"].joint_names == wheel_names
    for term in (env.observations.policy.joint_pos, env.observations.policy.joint_vel):
        assert term.params["asset_cfg"].joint_names == leg_names + wheel_names
        assert term.params["asset_cfg"].preserve_order
    assert env.rewards.joint_pos_limits.params["asset_cfg"].joint_names == leg_names
    assert env.actions.joint_pos.clip == {".*": (-100.0, 100.0)}
    assert env.actions.joint_vel.clip == {".*": (-100.0, 100.0)}
    assert env.terminations.illegal_contact is None
    base = env.base_link_name
    assert base in {link.get("name") for link in robot.findall("link")}
    assert agent.experiment_name == f"e1_wheel_{terrain.lower()}"
    if terrain == "Flat":
        assert env.scene.terrain.terrain_type == "plane"
        assert env.scene.height_scanner is None
        assert env.rewards.base_height_l2 is None
        assert env.curriculum.terrain_levels is None
    else:
        assert env.scene.terrain.terrain_type == "generator"
        assert env.scene.height_scanner.prim_path.endswith(f"/Geometry/{base}")
        assert env.scene.height_scanner_base.prim_path.endswith(f"/Geometry/{base}")
        assert env.curriculum.terrain_levels is not None


def test_cli_discovers_e1_wheel_tasks_without_manual_import():
    result = subprocess.run(
        [sys.executable, "-m", "isaaclab", "list_envs", "--keyword", "E1-Wheel"],
        capture_output=True,
        text=True,
        check=True,
    )
    for terrain in ("Flat", "Rough"):
        assert f"RobotLab-Isaac-Velocity-{terrain}-E1-Wheel-v0" in result.stdout


@pytest.mark.parametrize("terrain", ["Flat", "Rough"])
def test_e1_wheel_training_settings_match_go2w(terrain):
    from isaaclab_tasks.utils import resolve_task_config

    import robot_lab.tasks  # noqa: F401

    env, agent = resolve_task_config(
        f"RobotLab-Isaac-Velocity-{terrain}-E1-Wheel-v0", "rsl_rl_cfg_entry_point", overrides=[]
    )
    go2w, go2w_agent = resolve_task_config(
        f"RobotLab-Isaac-Velocity-{terrain}-Unitree-Go2W-v0", "rsl_rl_cfg_entry_point", overrides=[]
    )

    def without_robot_selectors(value):
        # Robot-specific selections have different lengths and names; compare the training parameters.
        if isinstance(value, dict):
            return {
                key: without_robot_selectors(item)
                for key, item in value.items()
                if key not in {"joint_names", "body_names", "mirror_joints", "synced_feet_pair_names"}
            }
        if isinstance(value, (list, tuple)):
            return [without_robot_selectors(item) for item in value]
        return value

    for section in ("commands", "events", "rewards", "terminations", "curriculum", "observations"):
        assert without_robot_selectors(getattr(env, section).to_dict()) == without_robot_selectors(
            getattr(go2w, section).to_dict()
        ), section
    assert env.scene.num_envs == go2w.scene.num_envs
    assert env.scene.terrain.to_dict() == go2w.scene.terrain.to_dict()
    assert env.sim.to_dict() == go2w.sim.to_dict()
    assert env.decimation == go2w.decimation
    assert env.episode_length_s == go2w.episode_length_s
    agent_values, go2w_agent_values = agent.to_dict(), go2w_agent.to_dict()
    agent_values.pop("experiment_name")
    go2w_agent_values.pop("experiment_name")
    assert agent_values == go2w_agent_values

    # Every E1 position joint must match exactly one scale pattern.
    for name in env.leg_joint_names:
        scales = [value for pattern, value in env.actions.joint_pos.scale.items() if re.fullmatch(pattern, name)]
        assert scales == [0.125 if "_hip_" in name else 0.25]
    assert env.actions.joint_vel.scale == go2w.actions.joint_vel.scale == 5.0


def test_e1_wheels_are_continuous_in_urdf():
    from robot_lab.assets.e1_wheel import E1_WHEEL_CFG

    robot = ET.parse(E1_WHEEL_CFG.spawn.asset_path).getroot()
    wheels = [joint for joint in robot.findall("joint") if joint.get("name", "").endswith("_wheel_joint")]
    assert len(wheels) == 4
    assert all(joint.get("type") == "continuous" for joint in wheels)
