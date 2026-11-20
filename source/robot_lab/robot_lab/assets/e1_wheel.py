from isaaclab_newton.sim.schemas import NewtonArticulationCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robot_lab.assets import ROBOT_LAB_ASSETS_DIR


E1_WHEEL_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        func=sim_utils.spawn_from_urdf,
        asset_path=f"{ROBOT_LAB_ASSETS_DIR}/Robots/E1_DOG_v2/E1_DOG_wheel_no_head.urdf",
        fix_base=False,
        merge_fixed_joints=False,
        # Preserve the authored collision geometry, including intentionally omitted colliders.
        collision_from_visuals=False,
        collision_type="Convex Hull",
        activate_contact_sensors=True,
        run_asset_transformer=True,
        run_multi_physics_conversion=True,
        rigid_props=PhysxRigidBodyCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=[
            PhysxArticulationCfg(
                enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=1
            ),
            NewtonArticulationCfg(self_collision_enabled=False),
        ],
        # Runtime actuators below supply the gains instead of the importer's defaults.
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.46),
        # Angles are in radians. Keep all four wheels at the same initial height.
        joint_pos={
            ".*_thigh_joint": 0.5,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_yaw_joint": 0.0,
            ".*_calf_joint": -1.0,
            ".*_wheel_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_thigh_joint", ".*_hip_roll_joint", ".*_hip_yaw_joint", ".*_calf_joint"],
            joint_effort_limit={
                ".*_thigh_joint": 60.0,
                ".*_hip_roll_joint": 36.0,
                ".*_hip_yaw_joint": 15.0,
                ".*_calf_joint": 60.0,
            },
            joint_velocity_limit={
                ".*_thigh_joint": 13.09,
                ".*_hip_roll_joint": 13.61,
                ".*_hip_yaw_joint": 15.71,
                ".*_calf_joint": 13.09,
            },
            stiffness={
                ".*_thigh_joint": 60.0,
                ".*_hip_roll_joint": 60.0,
                ".*_hip_yaw_joint": 30.0,
                ".*_calf_joint": 60.0,
            },
            damping={
                ".*_thigh_joint": 3.0,
                ".*_hip_roll_joint": 3.0,
                ".*_hip_yaw_joint": 1.5,
                ".*_calf_joint": 3.0,
            },
            friction=0.0,
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_wheel_joint"],
            joint_effort_limit=60.0,
            joint_velocity_limit=13.09,
            stiffness=0.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)
