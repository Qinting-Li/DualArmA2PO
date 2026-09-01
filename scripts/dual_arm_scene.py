#!/usr/bin/env python3
"""Isaac Lab smoke scene for a dual-Franka peg-in-hole experiment.

This validates asset loading, collision geometry, and contact sensing. It does
not train policies. Run it only from an Isaac Lab Python 3.12 environment.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Dual-Franka peg-in-hole smoke scene")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG


def fixed_block(size: tuple[float, float, float], pos: tuple[float, float, float]):
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/HoleFixture/Block",
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.25, 0.28, 0.32), metallic=0.6
            ),
            activate_contact_sensors=True,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )


@configclass
class BimanualInsertionSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg()
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8)),
    )
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 1.2, 0.20),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.30, 0.22)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.10)),
    )

    left_arm = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/LeftArm",
        init_state=FRANKA_PANDA_HIGH_PD_CFG.init_state.replace(
            pos=(0.0, -0.55, 0.20), rot=(0.7071068, 0.0, 0.0, 0.7071068)
        ),
    )
    right_arm = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/RightArm",
        init_state=FRANKA_PANDA_HIGH_PD_CFG.init_state.replace(
            pos=(0.0, 0.55, 0.20), rot=(0.7071068, 0.0, 0.0, -0.7071068)
        ),
    )

    peg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Peg",
        spawn=sim_utils.CylinderCfg(
            radius=0.020,
            height=0.180,
            axis="Z",
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.001, rest_offset=0.0
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_linear_velocity=1.0,
                max_angular_velocity=2.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.30),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.25, dynamic_friction=0.20, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.72, 0.18, 0.12), metallic=0.5
            ),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0015, 0.0, 0.35)),
    )

    # Four blocks leave a 44 mm square opening around the 40 mm diameter peg.
    hole_left = fixed_block((0.038, 0.120, 0.040), (-0.041, 0.0, 0.22)).replace(
        prim_path="{ENV_REGEX_NS}/HoleFixture/Left"
    )
    hole_right = fixed_block((0.038, 0.120, 0.040), (0.041, 0.0, 0.22)).replace(
        prim_path="{ENV_REGEX_NS}/HoleFixture/Right"
    )
    hole_front = fixed_block((0.044, 0.038, 0.040), (0.0, -0.041, 0.22)).replace(
        prim_path="{ENV_REGEX_NS}/HoleFixture/Front"
    )
    hole_back = fixed_block((0.044, 0.038, 0.040), (0.0, 0.041, 0.22)).replace(
        prim_path="{ENV_REGEX_NS}/HoleFixture/Back"
    )

    peg_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Peg",
        update_period=0.0,
        history_length=4,
        debug_vis=False,
    )


def hold_default_pose(scene: InteractiveScene) -> None:
    for name in ("left_arm", "right_arm"):
        robot = scene[name]
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = torch.zeros_like(joint_pos)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        robot.set_joint_position_target(joint_pos)


def main() -> None:
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    )
    sim.set_camera_view(eye=(1.4, 1.4, 1.2), target=(0.0, 0.0, 0.30))
    scene = InteractiveScene(
        BimanualInsertionSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    )
    sim.reset()
    hold_default_pose(scene)
    print("[INFO] Scene ready: two Frankas, peg, four-wall hole, contact sensor")

    step = 0
    try:
        while simulation_app.is_running():
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())
            if step % 200 == 0:
                forces = scene["peg_contact"].data.net_forces_w
                print(f"[INFO] step={step} max_contact_N={forces.norm(dim=-1).max().item():.3f}")
            step += 1
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
