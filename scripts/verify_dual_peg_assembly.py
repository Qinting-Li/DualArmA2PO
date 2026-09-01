#!/usr/bin/env python3
"""Verify scripted cooperative dual-peg insertion with two real Panda arms.

This is a baseline verification, not RL training.  It exercises the exact
Agent 1 -> Agent 2 -> Cartesian impedance -> dual-arm torque pipeline and
writes one row per control step plus plots of the policy-selected gains.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import x_bimanual.panda_dual_assembly as assembly  # noqa: E402


def scripted_actions(env: assembly.DualPandaAssemblyEnv, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A conservative object-pose baseline used before either RL policy."""
    translation_error = target[:3] - env.desired_pose[:3]
    rotation_error = assembly._rotvec(assembly._quat_mul(assembly._quat_conj(env.desired_pose[3:]), target[3:]))
    trajectory = np.r_[
        np.clip(translation_error / env.cfg.action_translation_limit, -1.0, 1.0),
        np.clip(rotation_error / env.cfg.action_rotation_limit_rad, -1.0, 1.0),
    ]
    # Agent 2 remains an action-producing policy.  The baseline lowers
    # lateral/rotational stiffness after contact while retaining axial support.
    if env.stage in (assembly.AssemblyStage.FIRST_CONTACT, assembly.AssemblyStage.COMPLIANT_ALIGNMENT, assembly.AssemblyStage.INSERTION):
        impedance = np.array([0.18, 0.18, 0.85, 0.15, 0.15, 0.15, 0.90, 0.90, 0.95, 0.95, 0.95, 0.95])
    else:
        impedance = np.array([0.80, 0.80, 0.90, 0.75, 0.75, 0.75, 0.75, 0.75, 0.80, 0.80, 0.80, 0.80])
    return trajectory, impedance


def write_log(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "step", "assembly_stage", "stage_index", "contact", "jamming", "success",
        "dx", "dy", "dz", "droll", "dpitch", "dyaw",
        "Kx", "Ky", "Kz", "Krx", "Kry", "Krz", "Dx", "Dy", "Dz", "Drx", "Dry", "Drz",
        "Fx", "Fy", "Fz", "Tx", "Ty", "Tz", "peg1_lateral_error", "peg2_lateral_error",
        "peg1_insertion_depth", "peg2_insertion_depth", "relative_position_error", "relative_orientation_error",
        "agent1_reward", "agent2_reward",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def make_plots(rows: list[dict], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    steps = np.arange(len(rows))
    kp = np.asarray([[r[f"K{axis}"] for axis in ("x", "y", "z")] for r in rows])
    kr = np.asarray([[r[f"Kr{axis}"] for axis in ("x", "y", "z")] for r in rows])
    kd = np.asarray([[r[f"D{axis}"] for axis in ("x", "y", "z")] for r in rows])
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(steps, kp, label=["Kx", "Ky", "Kz"])
    ax[0].plot(steps, kr, linestyle="--", label=["Krx", "Kry", "Krz"])
    ax[0].set_ylabel("stiffness")
    ax[0].legend(ncol=3)
    ax[0].grid(alpha=0.25)
    ax[1].plot(steps, kd, label=["Dx", "Dy", "Dz"])
    ax[1].set_ylabel("damping")
    ax[1].set_xlabel("control step")
    ax[1].legend(ncol=3)
    ax[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(directory / "dual_peg_impedance_parameters.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, [r["peg1_insertion_depth"] for r in rows], label="peg1 depth")
    ax.plot(steps, [r["peg2_insertion_depth"] for r in rows], label="peg2 depth")
    ax.plot(steps, [r["peg1_lateral_error"] for r in rows], label="peg1 lateral")
    ax.plot(steps, [r["peg2_lateral_error"] for r in rows], label="peg2 lateral")
    ax.set_xlabel("control step")
    ax.set_ylabel("metres")
    ax.grid(alpha=0.25)
    ax.legend(ncol=4)
    fig.tight_layout()
    fig.savefig(directory / "dual_peg_alignment_and_depth.png", dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/task.yaml")
    parser.add_argument("--log", type=Path, default=ROOT / "results/dual_peg_assembly_log.csv")
    parser.add_argument("--figures", type=Path, default=ROOT / "figures/dual_peg_assembly")
    args = parser.parse_args()

    with args.config.open() as handle:
        raw_config = yaml.safe_load(handle)
    config = assembly.dual_config_from_mapping(raw_config)
    config = assembly.DualAssemblyConfig(**{**config.__dict__, "max_steps": args.max_steps})
    env = assembly.DualPandaAssemblyEnv(config, seed=args.seed)
    env.reset(args.seed)
    # Both peg tips must cross the receiver opening plane by at least the
    # configured stable insertion depth.
    target = np.array([0.0, 0.0, 0.20, 1.0, 0.0, 0.0, 0.0])
    rows: list[dict] = []
    for step in range(args.max_steps):
        trajectory, impedance = scripted_actions(env, target)
        _, reward, done, info = env.step(trajectory, impedance)
        wrench = np.asarray(info["wrench"])
        kp = np.asarray(info["kp"]); kd = np.asarray(info["kd"])
        row = {
            "step": step, "assembly_stage": info["stage"], "stage_index": info["stage_index"],
            "contact": int(info["contact"]), "jamming": int(info["jamming"]), "success": int(info["success"]),
            **dict(zip(("dx", "dy", "dz", "droll", "dpitch", "dyaw"), trajectory)),
            **dict(zip(("Kx", "Ky", "Kz", "Krx", "Kry", "Krz"), kp)),
            **dict(zip(("Dx", "Dy", "Dz", "Drx", "Dry", "Drz"), kd)),
            **dict(zip(("Fx", "Fy", "Fz", "Tx", "Ty", "Tz"), wrench)),
            "peg1_lateral_error": info["peg1_lateral_error"], "peg2_lateral_error": info["peg2_lateral_error"],
            "peg1_insertion_depth": info["peg1_depth"], "peg2_insertion_depth": info["peg2_depth"],
            "relative_position_error": info["relative_position_error"], "relative_orientation_error": info["relative_orientation_error"],
            "agent1_reward": info["agent1_reward"], "agent2_reward": info["agent2_reward"],
        }
        rows.append(row)
        if done:
            break
    write_log(rows, args.log)
    make_plots(rows, args.figures)
    lat, depth = env.peg_errors()
    passed = bool(env.success and np.all(lat < env.cfg.lateral_threshold) and np.all(depth > env.cfg.required_depth))
    print({"success": passed, "steps": len(rows), "stage": env.stage.name, "peg_lateral_error": lat.tolist(), "peg_insertion_depth": depth.tolist(), "orientation_error_deg": env._relative_orientation_error(), "log": str(args.log), "figures": str(args.figures)})
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
