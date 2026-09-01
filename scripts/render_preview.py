#!/usr/bin/env python3
"""Render a lightweight preview using the locally installed Panda URDF."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pybullet as p
import pybullet_data


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "franka_dual_arm_preview.png"


def add_box(half_extents, position, color):
    visual = p.createVisualShape(
        p.GEOM_BOX, halfExtents=half_extents, rgbaColor=(*color, 1.0)
    )
    return p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=visual,
        basePosition=position,
    )


def pose_robot(robot_id: int, arm_positions: list[float]) -> None:
    arm_index = 0
    for joint_id in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, joint_id)
        joint_type = info[2]
        joint_name = info[1].decode("utf-8")
        if joint_type == p.JOINT_REVOLUTE and arm_index < 7:
            p.resetJointState(robot_id, joint_id, arm_positions[arm_index])
            arm_index += 1
        elif "finger_joint" in joint_name:
            p.resetJointState(robot_id, joint_id, 0.035)


def main() -> None:
    client = p.connect(p.DIRECT)
    if client < 0:
        raise RuntimeError("Could not start PyBullet in DIRECT mode")
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setGravity(0.0, 0.0, -9.81)
        p.loadURDF("plane.urdf")

        add_box((0.70, 0.55, 0.04), (0.0, 0.0, 0.20), (0.18, 0.21, 0.24))
        panda_urdf = "franka_panda/panda.urdf"
        left = p.loadURDF(
            panda_urdf,
            basePosition=(-0.52, 0.0, 0.24),
            baseOrientation=p.getQuaternionFromEuler((0.0, 0.0, 0.0)),
            useFixedBase=True,
        )
        right = p.loadURDF(
            panda_urdf,
            basePosition=(0.52, 0.0, 0.24),
            baseOrientation=p.getQuaternionFromEuler((0.0, 0.0, np.pi)),
            useFixedBase=True,
        )
        pose_robot(left, [0.0, -0.55, 0.0, -1.95, 0.0, 1.45, 0.75])
        pose_robot(right, [0.0, -0.55, 0.0, -1.95, 0.0, 1.45, 0.75])

        # Central peg and four walls visualize the intended insertion fixture.
        peg_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.020,
            length=0.180,
            rgbaColor=(0.75, 0.12, 0.08, 1.0),
        )
        p.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=peg_visual,
            basePosition=(0.0, 0.0, 0.48),
        )
        fixture_color = (0.58, 0.60, 0.64)
        add_box((0.019, 0.060, 0.020), (-0.041, 0.0, 0.28), fixture_color)
        add_box((0.019, 0.060, 0.020), (0.041, 0.0, 0.28), fixture_color)
        add_box((0.022, 0.019, 0.020), (0.0, -0.041, 0.28), fixture_color)
        add_box((0.022, 0.019, 0.020), (0.0, 0.041, 0.28), fixture_color)

        view = p.computeViewMatrix(
            cameraEyePosition=(1.45, -1.65, 1.15),
            cameraTargetPosition=(0.0, 0.0, 0.55),
            cameraUpVector=(0.0, 0.0, 1.0),
        )
        projection = p.computeProjectionMatrixFOV(
            fov=43.0, aspect=1.5, nearVal=0.05, farVal=5.0
        )
        width, height, rgba, _, _ = p.getCameraImage(
            width=1200,
            height=800,
            viewMatrix=view,
            projectionMatrix=projection,
            renderer=p.ER_TINY_RENDERER,
            shadow=1,
            lightDirection=(-1.0, -1.0, -2.0),
        )
        image = Image.fromarray(np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4))
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        image.convert("RGB").save(OUTPUT, quality=95)
        print(OUTPUT)
    finally:
        p.disconnect()


if __name__ == "__main__":
    main()
