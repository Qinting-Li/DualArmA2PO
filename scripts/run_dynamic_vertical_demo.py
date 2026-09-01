#!/usr/bin/env python3
"""Run a physical scripted dual-Panda grasp, transport, dynamic alignment, and insertion demo."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from x_bimanual.panda_dynamic_vertical_assembly import DynamicVerticalDualPandaEnv  # noqa: E402
from x_bimanual.panda_dual_assembly import AssemblyStage, _mat_quat, _quat_conj, _quat_mul, _rotvec, dual_config_from_mapping  # noqa: E402


def site_position(env, arm: int) -> np.ndarray:
    return env.data.site_xpos[env.ee_ids[arm]].copy()


def position_ik(env, arm: int, target: np.ndarray, q0: np.ndarray) -> np.ndarray:
    """Position-only damped least-squares IK for the real Panda joints."""
    q = q0.copy()
    for _ in range(120):
        env.data.qpos[env.arm_qpos[arm]] = q
        mujoco.mj_forward(env.model, env.data)
        error = target - site_position(env, arm)
        if np.linalg.norm(error) < 1e-4: break
        jp = np.zeros((3, env.model.nv)); jr = np.zeros((3, env.model.nv))
        mujoco.mj_jacSite(env.model, env.data, jp, jr, int(env.ee_ids[arm]))
        jac = jp[:, env.arm_dof[arm]]
        dq = jac.T @ np.linalg.solve(jac @ jac.T + 0.002 * np.eye(3), error)
        q = np.clip(q + 0.65 * dq, env.model.jnt_range[env.arm_jids[arm], 0], env.model.jnt_range[env.arm_jids[arm], 1])
    return q


def arm_pd(env, q_targets: np.ndarray, steps: int, rows: list[dict], phase: str) -> None:
    kp, kd = 80.0, 4.0
    for _ in range(steps):
        for arm in range(2):
            q = env.data.qpos[env.arm_qpos[arm]]; qd = env.data.qvel[env.arm_dof[arm]]
            bias = env.data.qfrc_bias[env.arm_dof[arm]]
            env.data.ctrl[arm * 7:(arm + 1) * 7] = np.clip(kp * (q_targets[arm] - q) - kd * qd + bias, -np.asarray(env.cfg.max_joint_torque), np.asarray(env.cfg.max_joint_torque))
        env._receiver_disturbance()
        mujoco.mj_step(env.model, env.data)
        env.step_count += 1
        log_row(env, rows, phase, np.zeros(6), np.full(12, .7))


def set_weld_relpose(env: DynamicVerticalDualPandaEnv) -> None:
    """Capture the current arm/object relation before enabling each weld."""
    obj = env.workpiece_body
    obj_pos = env.data.xpos[obj].copy(); obj_mat = env.data.xmat[obj].reshape(3, 3)
    for eqid, arm_name in enumerate(("left_panda_link8", "right_panda_link8")):
        arm = env.model.body(arm_name).id; arm_pos = env.data.xpos[arm].copy(); arm_mat = env.data.xmat[arm].reshape(3, 3)
        env.model.eq_data[eqid, 3:6] = arm_mat.T @ (obj_pos - arm_pos)
        env.model.eq_data[eqid, 6:10] = _mat_quat(arm_mat.T @ obj_mat)


def log_row(env, rows: list[dict], phase: str, action: np.ndarray, impedance: np.ndarray) -> None:
    lat, dep = env.peg_errors(); wrench = env._contact_wrench(); receiver_pos = env.data.xpos[env.receiver_body]
    internal_force = env.cfg.internal_force_min + float(impedance[-1]) * (env.cfg.internal_force_max - env.cfg.internal_force_min) if impedance.size == 7 else 0.0
    rows.append({"step": env.step_count, "phase": phase, "stage": env.stage.name, "grasped": int(env.grasped), "success": int(env.success), "object_x": float(env._workpiece_pose()[0][0]), "object_y": float(env._workpiece_pose()[0][1]), "object_z": float(env._workpiece_pose()[0][2]), "object_quat": env._workpiece_pose()[1].tolist(), "receiver_x": float(receiver_pos[0]), "receiver_y": float(receiver_pos[1]), "receiver_z": float(receiver_pos[2]), "receiver_quat": env.data.xquat[env.receiver_body].tolist(), "left_q": env.data.qpos[env.arm_qpos[0]].tolist(), "right_q": env.data.qpos[env.arm_qpos[1]].tolist(), "left_ee": site_position(env, 0).tolist(), "right_ee": site_position(env, 1).tolist(), "Fx": float(wrench[0]), "Fy": float(wrench[1]), "Fz": float(wrench[2]), "force_N": float(np.linalg.norm(wrench[:3])), "internal_force_N": internal_force, "peg1_lateral": float(lat[0]), "peg2_lateral": float(lat[1]), "peg1_depth": float(dep[0]), "peg2_depth": float(dep[1]), "trajectory_action": action.tolist(), "impedance_action": impedance.tolist()})


def main() -> int:
    out = ROOT / "outputs/a2po_dual_panda_dynamic_vertical"; out.mkdir(parents=True, exist_ok=True)
    with (ROOT / "configs/task.yaml").open() as handle:
        cfg = dual_config_from_mapping(yaml.safe_load(handle))
    env = DynamicVerticalDualPandaEnv(cfg, seed=20260824); env.reset(20260824)
    rows: list[dict] = []
    object_pos = env._workpiece_pose()[0].copy()
    grasp_targets = np.array([object_pos + [-.115, 0, 0], object_pos + [.115, 0, 0]])
    q_targets = np.array([position_ik(env, arm, grasp_targets[arm], env.rest_q.copy()) for arm in range(2)])
    arm_pd(env, q_targets, 180, rows, "APPROACH_GRASP")
    # The object remains at its original free-body pose.  We only capture the
    # live arm/object relation and then activate the two physical welds.
    set_weld_relpose(env)
    # Remove approach transients before closing both physical welds.  The
    # object remains at its live free-body pose; no pose teleport is used.
    env.data.qvel[:] = 0.0
    env.data.ctrl[:] = 0.0
    mujoco.mj_forward(env.model, env.data)
    env.enable_cooperative_grasp()
    for _ in range(20):
        for arm in range(2): env.data.ctrl[arm * 7:(arm + 1) * 7] = env.data.qfrc_bias[env.arm_dof[arm]]
        log_row(env, rows, "GRASP", np.zeros(6), np.full(7, .75)); mujoco.mj_step(env.model, env.data); env.step_count += 1
    env.desired_pose = np.r_[env._workpiece_pose()[0], env._workpiece_pose()[1]]
    impedance = np.array([.50, .65, .65, .85, .90, .90, .65])
    for _ in range(600):
        target = env.control_target_pose()
        translation = (target[:3] - env.desired_pose[:3]) / env.cfg.action_translation_limit
        rotation = _rotvec(_quat_mul(_quat_conj(env.desired_pose[3:]), target[3:])) / env.cfg.action_rotation_limit_rad
        action = np.clip(np.r_[translation, rotation], -1.0, 1.0)
        _, _, done, info = env.step(action, impedance)
        log_row(env, rows, info.get("phase", env.phase), action, impedance)
        if done:
            break
    fields = list(rows[0]);
    with (out / "dynamic_vertical_demo.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    summary = {"steps": len(rows), "grasped": env.grasped, "grasp_step": env.grasp_step, "success": env.success, "final_stage": env.stage.name, "final_peg_errors": dict(zip(("peg1_lateral", "peg2_lateral"), env.peg_errors()[0].tolist())), "final_depths": env.peg_errors()[1].tolist(), "peak_force_N": max(row["force_N"] for row in rows), "physical_two_arm_ee_start": rows[0]["left_ee"], "physical_two_arm_ee_end": rows[-1]["left_ee"]}
    (out / "dynamic_vertical_demo_summary.json").write_text(json.dumps(summary, indent=2)); print(json.dumps(summary, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
