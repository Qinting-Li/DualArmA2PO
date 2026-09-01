from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from x_bimanual.mujoco_floating import (  # noqa: E402
    ExperimentMode,
    FloatingConfig,
    FloatingInsertionEnv,
    RelativeStage,
    target_displacement,
)


class FloatingMuJoCoTests(unittest.TestCase):
    def test_floating_target_has_freejoint_and_zero_gravity(self):
        env = FloatingInsertionEnv(FloatingConfig(), ExperimentMode.FLOATING_ZERO_VELOCITY)
        self.assertEqual(env.model.joint("target_free").type[()], 0)  # mjJNT_FREE
        np.testing.assert_allclose(env.model.opt.gravity, [0.0, 0.0, 0.0])
        self.assertEqual(env.model.nq, 20)

    def test_fixed_baseline_has_no_target_freejoint(self):
        env = FloatingInsertionEnv(FloatingConfig(), ExperimentMode.FIXED)
        with self.assertRaises((KeyError, ValueError)):
            env.model.joint("target_free")

    def test_target_moves_from_initial_velocity_without_scripted_reset(self):
        env = FloatingInsertionEnv(FloatingConfig(), ExperimentMode.FLOATING_RANDOM_VELOCITY, seed=11)
        env.reset(11)
        initial = env.target_pose()[0].copy()
        for _ in range(20):
            env.step(env.agent1_action())
        displacement, rotation = target_displacement(env)
        self.assertGreater(np.linalg.norm(env.target_pose()[0] - initial), 0.0)
        self.assertGreater(displacement, 0.0)
        self.assertGreaterEqual(rotation, 0.0)

    def test_contact_wrench_is_observable_and_stage_is_relative(self):
        env = FloatingInsertionEnv(FloatingConfig(), ExperimentMode.FLOATING_ZERO_VELOCITY, seed=100007)
        env.reset(100007)
        peak = 0.0
        for _ in range(100):
            state = env.step(env.agent1_action())
            peak = max(peak, float(np.linalg.norm(state["contact_force"])))
        self.assertGreater(peak, 0.0)
        self.assertIn(env.stage, set(RelativeStage))
        observation = env.agent1_observation()
        for key in ("relative_position", "relative_orientation", "relative_linear_velocity", "relative_angular_velocity", "target_position", "target_orientation", "target_linear_velocity", "target_angular_velocity", "insertion_depth", "contact_force", "contact_torque", "stage"):
            self.assertIn(key, observation)


if __name__ == "__main__":
    unittest.main()
