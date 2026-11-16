# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Check that discovered E1 tasks target the authored robot instead of the Go2W template."""

import math
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

    # Observations use the same ordering as the two action terms and exclude unbounded wheel angles.
    for group in (env.observations.policy, env.observations.critic):
        assert group.joint_pos.params["asset_cfg"].joint_names == leg_names
        assert group.joint_vel.params["asset_cfg"].joint_names == leg_names + wheel_names
        assert group.joint_pos.params["asset_cfg"].preserve_order
        assert group.joint_vel.params["asset_cfg"].preserve_order
    assert env.rewards.joint_pos_limits.params["asset_cfg"].joint_names == leg_names

    # Target clipping must use each E1 joint's actual position limits, after adding the default pose.
    for name in leg_names:
        matches = [bounds for pattern, bounds in env.actions.joint_pos.clip.items() if re.fullmatch(pattern, name)]
        assert len(matches) == 1
        limit = joints[name].find("limit")
        assert matches[0] == (float(limit.get("lower")), float(limit.get("upper")))
    for name in wheel_names:
        speed = float(joints[name].find("limit").get("velocity"))
        assert env.actions.joint_vel.clip[".*"] == (-speed, speed)

    base = env.terminations.illegal_contact.params["sensor_cfg"].body_names[0]
    assert base in {link.get("name") for link in robot.findall("link")}
    assert agent.experiment_name == f"e1_wheel_{terrain.lower()}"
    assert env.commands.base_velocity.ranges.lin_vel_y == (0.0, 0.0)
    if terrain == "Flat":
        assert env.scene.terrain.terrain_type == "plane"
        assert env.scene.height_scanner is None
        assert env.scene.height_scanner_base is None
        assert env.rewards.base_height_l2.params["sensor_cfg"] is None
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


def test_spawn_overrides_only_wheel_limits_without_editing_urdf(monkeypatch):
    from pxr import Usd, UsdGeom, UsdPhysics

    import robot_lab.assets.e1_wheel as asset

    stage = Usd.Stage.CreateInMemory()
    prim = UsdGeom.Xform.Define(stage, "/Robot").GetPrim()
    for leg in ("FR", "FL", "RR", "RL"):
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"/Robot/{leg}_wheel_joint")
        joint.CreateLowerLimitAttr(0.0)
        joint.CreateUpperLimitAttr(0.0)
    thigh = UsdPhysics.RevoluteJoint.Define(stage, "/Robot/FR_thigh_joint")
    thigh.CreateLowerLimitAttr(-180.0)
    thigh.CreateUpperLimitAttr(180.0)
    monkeypatch.setattr(asset.sim_utils, "spawn_from_urdf", lambda *args, **kwargs: prim)
    path = Path(asset.E1_WHEEL_CFG.spawn.asset_path)
    original = path.read_bytes()
    asset.spawn_e1_wheel.__wrapped__("/Robot", asset.E1_WHEEL_CFG.spawn)
    for leg in ("FR", "FL", "RR", "RL"):
        joint = UsdPhysics.RevoluteJoint.Get(stage, f"/Robot/{leg}_wheel_joint")
        assert joint.GetLowerLimitAttr().Get() == -math.inf
        assert joint.GetUpperLimitAttr().Get() == math.inf
    assert thigh.GetLowerLimitAttr().Get() == -180.0
    assert thigh.GetUpperLimitAttr().Get() == 180.0
    assert path.read_bytes() == original
