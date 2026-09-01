from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from x_bimanual.osc import (
    BimanualOperationalSpaceMapper,
    OperationalSpaceLimits,
    make_default_osc_mapper,
)


class OperationalSpaceMapperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mapper = make_default_osc_mapper()
        self.jacobian = np.zeros((2, 6, 7))
        self.jacobian[:, :, :6] = np.eye(6)
        self.zeros7 = np.zeros((2, 7))

    def compute(self, **overrides):
        values = {
            "wrench": np.zeros((2, 6)),
            "jacobian": self.jacobian,
            "joint_position": self.zeros7,
            "joint_velocity": self.zeros7,
            "rest_position": self.zeros7,
            "stopped": False,
        }
        values.update(overrides)
        return self.mapper.compute(**values)

    def test_maps_each_arm_wrench_with_jacobian_transpose(self):
        wrench = np.array(
            [[1, 2, 3, 4, 5, 6], [-1, -2, -3, -4, -5, -6]], dtype=float
        )
        output = self.compute(wrench=wrench)
        np.testing.assert_allclose(output.task_torque[:, :6], wrench)
        np.testing.assert_allclose(output.task_torque[:, 6], 0.0)
        np.testing.assert_allclose(output.joint_torque, output.task_torque)

    def test_regulates_redundant_joint_in_nullspace(self):
        rest = self.zeros7.copy()
        rest[:, 6] = [0.2, -0.3]
        output = self.compute(rest_position=rest)
        np.testing.assert_allclose(output.nullspace_torque[:, :6], 0.0, atol=1e-12)
        np.testing.assert_allclose(output.nullspace_torque[:, 6], [2.0, -3.0])

    def test_clips_each_joint_and_reports_saturation(self):
        mapper = BimanualOperationalSpaceMapper(
            OperationalSpaceLimits(max_joint_torque=np.ones(7), nullspace_kp=0.0)
        )
        output = mapper.compute(
            wrench=np.full((2, 6), 3.0),
            jacobian=self.jacobian,
            joint_position=self.zeros7,
            joint_velocity=self.zeros7,
            rest_position=self.zeros7,
            feedforward_torque=np.full((2, 7), 2.0),
            stopped=False,
        )
        np.testing.assert_allclose(output.joint_torque, 1.0)
        self.assertTrue(np.all(output.saturated))

    def test_safety_stop_zeros_all_torque_sources(self):
        output = self.compute(
            wrench=np.ones((2, 6)),
            feedforward_torque=np.ones((2, 7)),
            stopped=True,
        )
        np.testing.assert_allclose(output.joint_torque, 0.0)
        np.testing.assert_allclose(output.task_torque, 0.0)
        np.testing.assert_allclose(output.nullspace_torque, 0.0)

    def test_rejects_wrong_jacobian_shape(self):
        with self.assertRaises(ValueError):
            self.compute(jacobian=np.zeros((2, 7, 6)))

    def test_rank_deficient_jacobian_remains_finite(self):
        output = self.compute(
            wrench=np.ones((2, 6)), jacobian=np.zeros((2, 6, 7))
        )
        self.assertTrue(np.all(np.isfinite(output.joint_torque)))
        np.testing.assert_allclose(output.task_torque, 0.0)


if __name__ == "__main__":
    unittest.main()
