from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from x_bimanual.controller import make_default_controller


class ControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = make_default_controller()
        self.zeros6 = np.zeros((2, 6))
        self.zeros3 = np.zeros((2, 3))

    def run_step(self, **overrides):
        values = {
            "pose_error": self.zeros6,
            "twist": self.zeros6,
            "trajectory_correction": self.zeros6,
            "kp_scale": np.full((2, 6), 0.5),
            "damping_ratio_scale": np.full((2, 6), 0.5),
            "contact_force": self.zeros3,
            "sync_error": 0.0,
        }
        values.update(overrides)
        return self.controller.step(**values)

    def test_zero_error_produces_zero_wrench(self):
        output = self.run_step()
        np.testing.assert_allclose(output.wrench, 0.0)
        self.assertFalse(output.stopped)

    def test_wrench_uses_vector_norm_limits(self):
        output = self.run_step(pose_error=np.full((2, 6), 100.0))
        np.testing.assert_allclose(np.linalg.norm(output.wrench[:, :3], axis=1), 80.0)
        np.testing.assert_allclose(np.linalg.norm(output.wrench[:, 3:], axis=1), 12.0)

    def test_excess_contact_force_stops_both_arms(self):
        forces = np.array([[46.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        output = self.run_step(contact_force=forces)
        self.assertTrue(output.stopped)
        self.assertEqual(output.stop_reason, "contact_force_limit")
        np.testing.assert_allclose(output.wrench, 0.0)

    def test_rejects_wrong_action_shape(self):
        with self.assertRaises(ValueError):
            self.run_step(kp_scale=np.zeros(6))


if __name__ == "__main__":
    unittest.main()

