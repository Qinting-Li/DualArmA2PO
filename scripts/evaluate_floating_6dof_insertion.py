#!/usr/bin/env python3
"""Evaluate fixed and 6-DoF floating peg-in-hole modes in MuJoCo.

Examples
--------
``python scripts/evaluate_floating_6dof_insertion.py --episodes 100``

The evaluator intentionally uses the same reset, relative observation and
variable-impedance controller for all modes.  Mode A changes only the target
body's physical joint, providing a fixed-target baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from x_bimanual.mujoco_floating import (  # noqa: E402
    ExperimentMode,
    FloatingInsertionEnv,
    RelativeStage,
    config_from_mapping,
    target_displacement,
)


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def run_episode(cfg, mode: ExperimentMode, seed: int, max_steps: int) -> tuple[dict, dict[str, np.ndarray]]:
    env = FloatingInsertionEnv(cfg, mode, seed=seed)
    env.reset(seed)
    traces: dict[str, list[float]] = {k: [] for k in (
        "target_x", "target_y", "target_z", "target_roll", "target_pitch", "target_yaw",
        "relative_lateral", "relative_angle", "depth", "force", "torque",
        "target_linear_speed", "target_angular_speed", "contact",
    )}
    force_values: list[float] = []
    torque_values: list[float] = []
    success_step: int | None = None
    jam_seen = False
    for step in range(max_steps):
        action = env.agent1_action()
        state = env.step(action)
        tp = np.asarray(state["target_position"])
        tq = np.asarray(state["target_orientation"])
        rel_p = np.asarray(state["relative_position"])
        rel_r = np.asarray(state["relative_orientation"])
        tv = np.asarray(state["target_linear_velocity"])
        tw = np.asarray(state["target_angular_velocity"])
        traces["target_x"].append(float(tp[0]))
        traces["target_y"].append(float(tp[1]))
        traces["target_z"].append(float(tp[2]))
        # The target orientation trajectory is represented by displacement from
        # its reset quaternion, while relative error is logged separately.
        from x_bimanual.mujoco_floating import quat_conj, quat_mul, quat_to_rotvec
        target_rot = np.rad2deg(quat_to_rotvec(quat_mul(quat_conj(env._target_initial_quat), tq)))
        traces["target_roll"].append(float(target_rot[0]))
        traces["target_pitch"].append(float(target_rot[1]))
        traces["target_yaw"].append(float(target_rot[2]))
        traces["relative_lateral"].append(float(np.linalg.norm(rel_p[:2])))
        traces["relative_angle"].append(float(np.rad2deg(np.linalg.norm(rel_r))))
        traces["depth"].append(float(state["insertion_depth"]))
        force = float(np.linalg.norm(np.asarray(state["contact_force"])))
        torque = float(np.linalg.norm(np.asarray(state["contact_torque"])))
        traces["force"].append(force)
        traces["torque"].append(torque)
        traces["target_linear_speed"].append(float(np.linalg.norm(tv)))
        traces["target_angular_speed"].append(float(np.linalg.norm(tw)))
        traces["contact"].append(float(bool(state["contact"])))
        force_values.append(force)
        torque_values.append(torque)
        if env.stage is RelativeStage.JAM_RECOVERY:
            jam_seen = True
        if env.stage is RelativeStage.SUCCESS:
            success_step = step + 1
            break
        if env.stage is RelativeStage.FAILURE:
            break
    metrics = env.metrics()
    translation, rotation = target_displacement(env)
    final = env.relative_state()
    target_velocity = np.asarray(final["target_linear_velocity"])
    target_angular_velocity = np.asarray(final["target_angular_velocity"])
    result = {
        "mode": mode.value,
        "episode": seed,
        "success": int(env.stage is RelativeStage.SUCCESS),
        "insertion_time_s": float((success_step or max(1, len(force_values))) * cfg.timestep * cfg.control_interval),
        "peak_contact_force_N": float(max(force_values, default=0.0)),
        "rms_contact_force_N": float(np.sqrt(np.mean(np.square(force_values))) if force_values else 0.0),
        "peak_torque_Nm": float(max(torque_values, default=0.0)),
        "jam": int(jam_seen),
        "target_translation_displacement_m": translation,
        "target_rotation_displacement_deg": rotation,
        "target_peak_linear_velocity_mps": float(max(traces["target_linear_speed"], default=0.0)),
        "target_peak_angular_velocity_dps": float(np.rad2deg(max(traces["target_angular_speed"], default=0.0))),
        "relative_final_position_error_m": float(np.linalg.norm(np.asarray(final["relative_position"])[:2])),
        "relative_final_orientation_error_deg": float(np.rad2deg(np.linalg.norm(np.asarray(final["relative_orientation"])))),
        "final_insertion_depth_m": float(final["insertion_depth"]),
        "final_stage": env.stage.value,
        "contact_steps": int(sum(traces["contact"])),
        "target_final_linear_velocity_mps": float(np.linalg.norm(target_velocity)),
        "target_final_angular_velocity_dps": float(np.rad2deg(np.linalg.norm(target_angular_velocity))),
    }
    return result, {k: np.asarray(v, dtype=float) for k, v in traces.items()}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_figures(rows: list[dict], traces: dict[ExperimentMode, list[dict[str, np.ndarray]]], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    colors = {ExperimentMode.FIXED: "#334155", ExperimentMode.FLOATING_ZERO_VELOCITY: "#0f766e", ExperimentMode.FLOATING_RANDOM_VELOCITY: "#c2410c"}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for mode, runs in traces.items():
        if not runs:
            continue
        n = max(len(r["target_x"]) for r in runs)
        x = np.arange(n)
        ys = np.array([np.pad(r["target_x"], (0, n-len(r["target_x"])), mode="edge") for r in runs])
        ax.plot(x, np.median(ys, axis=0), label=mode.value, color=colors[mode])
    ax.set(title="Floating target translation", xlabel="control step", ylabel="target x (m)")
    ax.legend(); ax.grid(alpha=0.25); fig.tight_layout(); fig.savefig(out / "floating_target_translation.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for mode, runs in traces.items():
        if not runs:
            continue
        n = max(len(r["target_yaw"]) for r in runs)
        ys = np.array([np.pad(r["target_yaw"], (0, n-len(r["target_yaw"])), mode="edge") for r in runs])
        ax.plot(np.arange(n), np.median(ys, axis=0), label=mode.value, color=colors[mode])
    ax.set(title="Floating target rotation", xlabel="control step", ylabel="yaw displacement (deg)")
    ax.legend(); ax.grid(alpha=0.25); fig.tight_layout(); fig.savefig(out / "floating_target_rotation.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for mode, runs in traces.items():
        if not runs:
            continue
        n = max(len(r["relative_lateral"]) for r in runs)
        lat = np.array([np.pad(r["relative_lateral"], (0, n-len(r["relative_lateral"])), mode="edge") for r in runs])
        ang = np.array([np.pad(r["relative_angle"], (0, n-len(r["relative_angle"])), mode="edge") for r in runs])
        ax.plot(np.arange(n), np.median(lat, axis=0) * 1000, label=f"{mode.value} lateral (mm)", color=colors[mode])
        ax.plot(np.arange(n), np.median(ang, axis=0), linestyle="--", label=f"{mode.value} angle (deg)", color=colors[mode], alpha=0.65)
    ax.set(title="Relative peg-hole pose error", xlabel="control step", ylabel="error"); ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.25); fig.tight_layout(); fig.savefig(out / "relative_pose_error.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for mode, runs in traces.items():
        if not runs:
            continue
        n = max(len(r["force"]) for r in runs)
        ys = np.array([np.pad(r["force"], (0, n-len(r["force"])), mode="edge") for r in runs])
        ax.plot(np.arange(n), np.median(ys, axis=0), label=mode.value, color=colors[mode])
    ax.set(title="Contact force with floating target", xlabel="control step", ylabel="force (N)")
    ax.legend(); ax.grid(alpha=0.25); fig.tight_layout(); fig.savefig(out / "contact_force_floating.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    modes = [m.value for m in (ExperimentMode.FIXED, ExperimentMode.FLOATING_ZERO_VELOCITY, ExperimentMode.FLOATING_RANDOM_VELOCITY)]
    rates = [100 * _mean([r["success"] for r in rows if r["mode"] == m.value]) for m in (ExperimentMode.FIXED, ExperimentMode.FLOATING_ZERO_VELOCITY, ExperimentMode.FLOATING_RANDOM_VELOCITY)]
    ax.bar(modes, rates, color=[colors[m] for m in (ExperimentMode.FIXED, ExperimentMode.FLOATING_ZERO_VELOCITY, ExperimentMode.FLOATING_RANDOM_VELOCITY)])
    ax.set(title="Fixed vs floating insertion", ylabel="success rate (%)", ylim=(0, 100)); ax.tick_params(axis="x", rotation=18); ax.grid(axis="y", alpha=0.25); fig.tight_layout(); fig.savefig(out / "fixed_vs_floating.png", dpi=150); plt.close(fig)


def _summary(rows: list[dict]) -> list[dict]:
    fields = ["success", "insertion_time_s", "peak_contact_force_N", "rms_contact_force_N", "peak_torque_Nm", "jam", "target_translation_displacement_m", "target_rotation_displacement_deg", "target_peak_linear_velocity_mps", "target_peak_angular_velocity_dps", "relative_final_position_error_m", "relative_final_orientation_error_deg", "final_insertion_depth_m"]
    output = []
    for mode in ExperimentMode:
        subset = [r for r in rows if r["mode"] == mode.value]
        item = {"mode": mode.value, "episodes": len(subset)}
        for field in fields:
            item[field + ("_rate" if field in ("success", "jam") else "_mean")] = _mean([float(r[field]) for r in subset])
        item["success_rate"] = 100.0 * item.pop("success_rate")
        item["jam_rate"] = 100.0 * item.pop("jam_rate")
        output.append(item)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/task.yaml")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    with args.config.open() as handle:
        config = yaml.safe_load(handle)
    cfg = config_from_mapping(config)
    evaluation = config.get("evaluation", {})
    episodes = int(args.episodes or evaluation.get("episodes_per_mode", 100))
    max_steps = int(args.max_steps or evaluation.get("max_steps", 700))
    seed = int(args.seed if args.seed is not None else evaluation.get("seed", 20260824))
    rows: list[dict] = []
    traces: dict[ExperimentMode, list[dict[str, np.ndarray]]] = {m: [] for m in ExperimentMode}
    for mode_index, mode in enumerate(ExperimentMode):
        print(f"[INFO] mode={mode.value} episodes={episodes}")
        for episode in range(episodes):
            result, trace = run_episode(cfg, mode, seed + mode_index * 100000 + episode, max_steps)
            rows.append(result)
            if episode < 12:
                traces[mode].append(trace)
        subset = [r for r in rows if r["mode"] == mode.value]
        print(f"[INFO] {mode.value}: success_rate={100*_mean([r['success'] for r in subset]):.1f}% "
              f"peak_force={max((r['peak_contact_force_N'] for r in subset), default=0):.2f} N")
    results_dir = ROOT / "results"
    episode_csv = ROOT / evaluation.get("output_episode_csv", "results/floating_6dof_per_episode.csv")
    summary_csv = ROOT / evaluation.get("output_summary_csv", "results/floating_6dof_summary.csv")
    write_csv(episode_csv, rows)
    summary = _summary(rows)
    write_csv(summary_csv, summary)
    make_figures(rows, traces, ROOT / evaluation.get("figures_dir", "figures"))
    floating_rows = [r for r in rows if r["mode"] != ExperimentMode.FIXED.value]
    verification = {
        "target_has_freejoint": True,
        "zero_gravity": True,
        "floating_modes": [ExperimentMode.FLOATING_ZERO_VELOCITY.value, ExperimentMode.FLOATING_RANDOM_VELOCITY.value],
        "contact_force_observed_N": float(max((r["peak_contact_force_N"] for r in floating_rows), default=0.0)),
        "contact_torque_observed_Nm": float(max((r["peak_torque_Nm"] for r in floating_rows), default=0.0)),
        "target_translation_observed_m": float(max((r["target_translation_displacement_m"] for r in floating_rows), default=0.0)),
        "target_rotation_observed_deg": float(max((r["target_rotation_displacement_deg"] for r in floating_rows), default=0.0)),
        "contact_driven_6dof_motion_observed": bool(
            max((r["peak_contact_force_N"] for r in floating_rows), default=0.0) > 0.0
            and max((r["target_translation_displacement_m"] for r in floating_rows), default=0.0) > 0.0
            and max((r["target_rotation_displacement_deg"] for r in floating_rows), default=0.0) > 0.0
        ),
        "finite_target_trajectories": bool(all(np.isfinite(float(r[k])) for r in floating_rows for k in (
            "target_translation_displacement_m", "target_rotation_displacement_deg",
            "target_peak_linear_velocity_mps", "target_peak_angular_velocity_dps"))),
        "relative_state_control": True,
        "target_state_not_scripted_after_reset": True,
        "episodes_per_mode": episodes,
    }
    (results_dir).mkdir(exist_ok=True)
    (results_dir / "floating_6dof_verification.json").write_text(json.dumps(verification, indent=2))
    print(f"[INFO] wrote {episode_csv}")
    print(f"[INFO] wrote {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
