from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from x_bimanual.a2po_training import A2POTrainConfig, A2POTrainer
from x_bimanual.panda_dual_assembly import DualAssemblyConfig, _quat_conj, _quat_mul, _rotvec
from x_bimanual.panda_dynamic_vertical_assembly import DynamicVerticalDualPandaEnv


class DynamicVerticalDualPandaTests(unittest.TestCase):
    def make_env(self, seed: int = 3) -> DynamicVerticalDualPandaEnv:
        cfg = replace(DualAssemblyConfig(gravity=(0.0, 0.0, 0.0)), max_steps=300)
        return DynamicVerticalDualPandaEnv(cfg, seed=seed)

    def test_dynamic_interface_and_trainer_action_width(self):
        env = self.make_env()
        observations = env.reset(3)
        self.assertEqual(observations["trajectory"].shape, (96,))
        self.assertEqual(observations["impedance"].shape, (115,))
        self.assertEqual(env.impedance_action_dim, 7)
        trainer = A2POTrainer(env, A2POTrainConfig(hidden_size=16), Path("/tmp/x-dynamic-test"))
        self.assertEqual(trainer.agent2.action_dim, 7)
        self.assertEqual(trainer._object_target_prior().shape, (6,))

    def test_both_arms_reach_and_enable_physical_grasp(self):
        env = self.make_env(5)
        env.reset(5)
        for _ in range(80):
            env.step(np.zeros(6), np.full(7, 0.5))
            if env.grasped:
                break
        self.assertTrue(env.grasped)
        self.assertTrue(np.all(env.data.eq_active == 1))
        self.assertLess(env.grasp_step, 80)
        self.assertLess(float(np.linalg.norm(env.data.efc_pos)), 5e-3)

    def test_internal_force_is_equal_opposite_and_bounded(self):
        env = self.make_env(6)
        env.reset(6)
        for _ in range(80):
            env.step(np.zeros(6), np.full(7, 0.5))
            if env.grasped:
                break
        internal = env._internal_force_wrench(23.0)
        np.testing.assert_allclose(internal[0, :3] + internal[1, :3], 0.0, atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.norm(internal[0, :3])), 23.0)
        action = np.array([0.50, 0.65, 0.65, 0.85, 0.90, 0.90, 0.25])
        _, _, _, info = env.step(np.zeros(6), action)
        expected = env.cfg.internal_force_min + 0.25 * (env.cfg.internal_force_max - env.cfg.internal_force_min)
        self.assertAlmostEqual(info["internal_force_N"], expected)
        self.assertAlmostEqual(info["grasp_capacity_N"], 2.0 * env.cfg.grasp_friction * expected)
        self.assertTrue(np.isfinite(info["grasp_margin_N"]))

    def test_persistent_negative_grasp_margin_releases_welds(self):
        env = self.make_env(8)
        env.reset(8)
        for _ in range(80):
            env.step(np.zeros(6), np.full(7, 0.5))
            if env.grasped:
                break
        for _ in range(env.cfg.grasp_slip_steps - 1):
            self.assertFalse(env._update_grasp_failure(-1.0))
        self.assertTrue(env._update_grasp_failure(-1.0))
        self.assertTrue(env.grasp_failed)
        self.assertFalse(env.grasped)
        self.assertTrue(np.all(env.data.eq_active == 0))

    def test_receiver_disturbance_is_bounded(self):
        env = self.make_env(7)
        env.reset(7)
        nominal = np.array([0.0, 0.12, 0.28])
        peak_displacement = 0.0
        env.data.ctrl[:] = 0.0
        for _ in range(600):
            env._receiver_disturbance()
            for _ in range(env.cfg.control_interval):
                mujoco.mj_step(env.model, env.data)
            env.step_count += 1
            peak_displacement = max(peak_displacement, float(np.linalg.norm(env.data.xpos[env.receiver_body] - nominal)))
        self.assertLess(peak_displacement, 0.03)

    def test_strict_scripted_prior_completes_dual_peg_insertion(self):
        cfg = replace(
            DualAssemblyConfig(gravity=(0.0, 0.0, 0.0)),
            lateral_threshold=0.001,
            required_depth=0.035,
            stable_steps=25,
            max_steps=500,
        )
        env = DynamicVerticalDualPandaEnv(cfg, seed=20260824)
        env.reset(20260824)
        impedance = np.array([0.50, 0.65, 0.65, 0.85, 0.90, 0.90, 0.65])
        for _ in range(cfg.max_steps):
            target = env.control_target_pose()
            translation = (target[:3] - env.desired_pose[:3]) / cfg.action_translation_limit
            rotation = _rotvec(_quat_mul(_quat_conj(env.desired_pose[3:]), target[3:])) / cfg.action_rotation_limit_rad
            _, _, done, _ = env.step(np.clip(np.r_[translation, rotation], -1.0, 1.0), impedance)
            if done:
                break
        lateral, depth = env.peg_errors()
        self.assertTrue(env.success)
        self.assertTrue(np.all(lateral < cfg.lateral_threshold))
        self.assertTrue(np.all(depth > cfg.required_depth))


if __name__ == "__main__":
    unittest.main()
