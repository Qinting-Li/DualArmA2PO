#!/usr/bin/env python3
"""Run the control core without Isaac Lab."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from x_bimanual.controller import make_default_controller
from x_bimanual.osc import make_default_osc_mapper


def main() -> None:
    controller = make_default_controller()
    output = controller.step(
        pose_error=np.array([[0.002, 0, 0, 0, 0, 0], [-0.002, 0, 0, 0, 0, 0]]),
        twist=np.zeros((2, 6)),
        trajectory_correction=np.zeros((2, 6)),
        kp_scale=np.full((2, 6), 0.5),
        damping_ratio_scale=np.full((2, 6), 0.5),
        contact_force=np.zeros((2, 3)),
        sync_error=0.0,
    )
    print("wrench [left, right]:")
    print(output.wrench)
    print("safety stop:", output.stopped)

    jacobian = np.zeros((2, 6, 7))
    jacobian[:, :, :6] = np.eye(6)
    osc_output = make_default_osc_mapper().compute(
        wrench=output.wrench,
        jacobian=jacobian,
        joint_position=np.zeros((2, 7)),
        joint_velocity=np.zeros((2, 7)),
        rest_position=np.zeros((2, 7)),
        stopped=output.stopped,
    )
    print("joint torque [left, right]:")
    print(osc_output.joint_torque)


if __name__ == "__main__":
    main()
