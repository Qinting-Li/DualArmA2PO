from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import x_bimanual.panda_dual_assembly as assembly


class PandaDualAssemblyTests(unittest.TestCase):
    def test_real_panda_model_and_shared_workpiece(self):
        env = assembly.DualPandaAssemblyEnv(seed=4)
        self.assertEqual(env.model.nu, 14)
        self.assertEqual(env.model.nq, 21)  # 14 Panda joints + 7-DoF free workpiece
        for side in ("left", "right"):
            for index in range(1, 8):
                self.assertGreaterEqual(env.model.joint(f"{side}_panda_joint{index}").id, 0)
        self.assertEqual(env.model.neq, 2)
        self.assertEqual(len(env.peg_ids), 2)
        self.assertEqual(len(env.hole_ids), 2)

    def test_observations_and_a2po_conditioning(self):
        env = assembly.DualPandaAssemblyEnv(seed=5)
        observations = env.reset(5)
        self.assertEqual(observations["trajectory"].shape, (30,))
        self.assertEqual(observations["impedance"].shape, (42,))
        coordinator = assembly.A2POCoordinator(lambda x: np.zeros(6), lambda x: np.zeros(12))
        actions = coordinator.act(observations)
        self.assertEqual(actions.trajectory.shape, (6,))
        self.assertEqual(actions.impedance.shape, (12,))

    def test_scripted_policy_inserts_both_pegs(self):
        env = assembly.DualPandaAssemblyEnv(seed=6)
        env.reset(6)
        target = np.array([0.0, 0.0, 0.20, 1.0, 0.0, 0.0, 0.0])
        for _ in range(100):
            delta = target[:3] - env.desired_pose[:3]
            rotation = assembly._rotvec(assembly._quat_mul(assembly._quat_conj(env.desired_pose[3:]), target[3:]))
            trajectory = np.r_[
                np.clip(delta / env.cfg.action_translation_limit, -1.0, 1.0),
                np.clip(rotation / env.cfg.action_rotation_limit_rad, -1.0, 1.0),
            ]
            _, _, done, _ = env.step(trajectory, np.ones(12))
            if done:
                break
        lateral, depth = env.peg_errors()
        self.assertTrue(env.success)
        self.assertTrue(np.all(lateral < env.cfg.lateral_threshold))
        self.assertTrue(np.all(depth > env.cfg.required_depth))
        self.assertLess(env._relative_orientation_error(), env.cfg.orientation_threshold_deg)


if __name__ == "__main__":
    unittest.main()
