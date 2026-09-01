#!/usr/bin/env python3
"""Frozen-protocol multi-seed A2PO training, evaluation, and reporting.

This script treats DynamicVerticalDualPandaEnv as read-only.  Training and
checkpoint selection use disjoint seed spaces; nominal, hard, and OOD test
seeds are never used for model selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from x_bimanual.a2po_training import (  # noqa: E402
    A2POTrainConfig,
    A2POTrainer,
    dynamic_impedance_prior,
)
from x_bimanual.panda_dual_assembly import (  # noqa: E402
    _quat_conj,
    _quat_mul,
    _rotvec,
    dual_config_from_mapping,
)
from x_bimanual.panda_dynamic_vertical_assembly import (  # noqa: E402
    DynamicVerticalDualPandaEnv,
)
from train_a2po_dynamic_vertical import curriculum  # noqa: E402

RESULT_ROOT = ROOT / "results/a2po_dual_panda_dynamic_vertical"
FORMAL_ROOT = RESULT_ROOT / "formal"
FIGURE_ROOT = ROOT / "figures/a2po_dual_panda_dynamic_vertical/formal"
VIDEO_ROOT = ROOT / "outputs/a2po_dual_panda_dynamic_vertical/formal_videos"
REGRESSION_XML = FORMAL_ROOT / "regression_tests.xml"

TRAIN_SEEDS = (20260901, 20260911, 20260921, 20260931, 20260941)
VALIDATION_SEEDS = tuple(range(40_000_000, 40_000_040))
EVALUATION_SEED_START = {
    "nominal": 41_000_000,
    "hard": 42_000_000,
    "ood": 43_000_000,
}
VARIANTS = (
    "controller_only",
    "a2po",
    "no_adaptive_impedance",
    "no_staged_alignment_recovery",
)


def load_base_config():
    with (ROOT / "configs/task.yaml").open() as handle:
        return replace(dual_config_from_mapping(yaml.safe_load(handle)), gravity=(0.0, 0.0, 0.0))


def evaluation_config(base, setting: str):
    if setting == "nominal":
        return base
    if setting == "hard":
        # Frozen to the pre-existing hard curriculum definition.
        return curriculum(base, 2, 3)
    if setting == "ood":
        return replace(
            base,
            initial_xy_range_m=0.012,
            initial_z_min_m=0.220,
            initial_z_max_m=0.340,
            initial_rotation_range_rad=0.080,
        )
    raise ValueError(f"unknown setting: {setting}")


def protocol_payload(base) -> dict[str, Any]:
    payload = {
        "environment": "DynamicVerticalDualPandaEnv",
        "environment_config": asdict(base),
        "train_seeds": list(TRAIN_SEEDS),
        "validation_seeds": list(VALIDATION_SEEDS),
        "evaluation_seed_start": EVALUATION_SEED_START,
        "hard_distribution": asdict(evaluation_config(base, "hard")),
        "ood_distribution": asdict(evaluation_config(base, "ood")),
        "checkpoint_selection": ["max_validation_success", "min_jam_rate", "min_peak_force", "min_mean_steps"],
        "policy_inputs": "current observations plus shared deterministic-controller prior action; no future state, success label, test seed, or test metric",
        "mappo_baseline": "not run: no reusable MAPPO implementation exists in the repository",
    }
    encoded = json.dumps(payload, sort_keys=True).encode()
    payload["protocol_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def frozen_artifact_hashes() -> dict[str, str]:
    paths = (
        "src/x_bimanual/panda_dynamic_vertical_assembly.py",
        "configs/task.yaml",
        "tests/test_panda_dynamic_vertical_assembly.py",
    )
    return {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in paths
    }


def regression_result() -> dict[str, Any]:
    root = ET.parse(REGRESSION_XML).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    if suite is None:
        raise ValueError(f"no testsuite found in {REGRESSION_XML}")
    result = {
        key: int(suite.attrib.get(key, 0))
        for key in ("tests", "failures", "errors", "skipped")
    }
    result["passed"] = result["tests"] - result["failures"] - result["errors"] - result["skipped"]
    result["all_passed"] = bool(result["tests"] == 23 and result["passed"] == 23)
    result["source"] = str(REGRESSION_XML.relative_to(ROOT))
    return result


def load_weights(trainer: A2POTrainer, checkpoint: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location=trainer.device)
    trainer.agent1.load_state_dict(payload["agent1"])
    trainer.agent2.load_state_dict(payload["agent2"])
    return payload


def target_action(env: DynamicVerticalDualPandaEnv, staged: bool) -> np.ndarray:
    target = env.control_target_pose() if staged else env.object_target_pose()
    translation = (target[:3] - env.desired_pose[:3]) / env.cfg.action_translation_limit
    rotation = _rotvec(_quat_mul(_quat_conj(env.desired_pose[3:]), target[3:])) / env.cfg.action_rotation_limit_rad
    return np.clip(np.r_[translation, rotation], -1.0, 1.0).astype(np.float32)


def policy_actions(trainer: A2POTrainer, obs: dict[str, np.ndarray], variant: str) -> tuple[np.ndarray, np.ndarray]:
    staged = variant != "no_staged_alignment_recovery"
    prior = target_action(trainer.env, staged=staged)
    if variant == "controller_only":
        return prior, dynamic_impedance_prior(np.empty(0))
    action1, _, _, _ = trainer.agent1.sample(obs["trajectory"], deterministic=True, prior_override=prior)
    if variant == "no_adaptive_impedance":
        action2 = dynamic_impedance_prior(np.empty(0))
    else:
        action2, _, _, _ = trainer.agent2.sample(np.r_[obs["impedance"], action1], deterministic=True)
    return action1, action2


def failure_reason(env: DynamicVerticalDualPandaEnv, any_jam: bool) -> str:
    if env.success:
        return "success"
    if env.grasp_failed:
        return "grasp_slip"
    lateral, depth = env.peg_errors()
    orientation = env._relative_orientation_error()
    if not env.grasped:
        return "grasp_timeout"
    if any_jam:
        return "jam"
    if np.max(depth) <= 0.001 and np.max(lateral) >= 0.012:
        return "coarse_alignment"
    if np.min(depth) <= env.cfg.required_depth:
        if np.max(lateral) >= env.cfg.lateral_threshold:
            return "contact_alignment"
        return "insufficient_depth"
    if np.max(lateral) >= env.cfg.lateral_threshold:
        return "lateral_tolerance"
    if orientation >= env.cfg.orientation_threshold_deg:
        return "orientation_tolerance"
    if env.stable_count < env.cfg.stable_steps:
        return "stabilization_timeout"
    return "timeout_other"


def rollout_episode(
    trainer: A2POTrainer,
    cfg,
    seed: int,
    variant: str,
    *,
    capture_trace: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env = DynamicVerticalDualPandaEnv(cfg, seed=seed)
    trainer.env = env
    obs = env.reset(seed)
    force_values: list[float] = []
    contact_force_values: list[float] = []
    torque_values: list[float] = []
    trajectory_residuals: list[float] = []
    impedance_residuals: list[float] = []
    impedance_actions: list[np.ndarray] = []
    any_jam = False
    contact_steps = 0
    trace: list[dict[str, Any]] = []
    for step in range(cfg.max_steps):
        action1, action2 = policy_actions(trainer, obs, variant)
        shared_prior = target_action(env, staged=variant != "no_staged_alignment_recovery")
        fixed_impedance = dynamic_impedance_prior(np.empty(0))
        trajectory_residuals.append(float(np.linalg.norm(action1 - shared_prior)))
        impedance_residuals.append(float(np.linalg.norm(action2 - fixed_impedance)))
        impedance_actions.append(np.asarray(action2, dtype=float).copy())
        obs, reward, done, info = env.step(action1, action2)
        wrench = np.asarray(info.get("wrench", np.zeros(6)), dtype=float)
        force = float(np.linalg.norm(wrench[:3]))
        torque = float(np.linalg.norm(wrench[3:]))
        force_values.append(force)
        torque_values.append(torque)
        if bool(info.get("contact", False)):
            contact_steps += 1
            contact_force_values.append(force)
        any_jam = any_jam or bool(info.get("jamming", False))
        if capture_trace:
            obj_pos, obj_quat = env._workpiece_pose()
            lateral_error, insertion_depth = env.peg_errors()
            trace.append({
                "step": step,
                "stage": info.get("stage", env.stage.name),
                "success": int(env.success),
                "object_pos": obj_pos.tolist(),
                "object_quat": obj_quat.tolist(),
                "receiver_pos": env.data.xpos[env.receiver_body].copy().tolist(),
                "receiver_quat": env.data.xquat[env.receiver_body].copy().tolist(),
                "left_q": env.data.qpos[env.arm_qpos[0]].copy().tolist(),
                "right_q": env.data.qpos[env.arm_qpos[1]].copy().tolist(),
                "force_N": force,
                "reward": float(reward),
                "peg1_lateral_error_m": float(lateral_error[0]),
                "peg2_lateral_error_m": float(lateral_error[1]),
                "peg1_depth_m": float(insertion_depth[0]),
                "peg2_depth_m": float(insertion_depth[1]),
                "orientation_error_deg": float(env._relative_orientation_error()),
                "trajectory_action": action1.tolist(),
                "trajectory_prior": shared_prior.tolist(),
                "impedance_action": action2.tolist(),
            })
        if done:
            break
    lateral, depth = env.peg_errors()
    row = {
        "seed": seed,
        "variant": variant,
        "success": int(env.success),
        "steps": step + 1,
        "completion_time_s": (step + 1) * cfg.timestep * cfg.control_interval,
        "peg1_depth_m": float(depth[0]),
        "peg2_depth_m": float(depth[1]),
        "mean_depth_m": float(np.mean(depth)),
        "min_depth_m": float(np.min(depth)),
        "peg1_lateral_error_m": float(lateral[0]),
        "peg2_lateral_error_m": float(lateral[1]),
        "mean_lateral_error_m": float(np.mean(lateral)),
        "max_lateral_error_m": float(np.max(lateral)),
        "orientation_error_deg": float(env._relative_orientation_error()),
        "peak_contact_force_N": float(max(force_values, default=0.0)),
        "mean_force_all_steps_N": float(np.mean(force_values)) if force_values else 0.0,
        "mean_force_contact_steps_N": float(np.mean(contact_force_values)) if contact_force_values else 0.0,
        "peak_contact_torque_Nm": float(max(torque_values, default=0.0)),
        "contact_steps": contact_steps,
        "mean_trajectory_residual_norm": float(np.mean(trajectory_residuals)),
        "peak_trajectory_residual_norm": float(np.max(trajectory_residuals)),
        "mean_impedance_residual_norm": float(np.mean(impedance_residuals)),
        "peak_impedance_residual_norm": float(np.max(impedance_residuals)),
        **{
            f"mean_impedance_{name}": float(np.mean(np.asarray(impedance_actions)[:, index]))
            for index, name in enumerate(("k_parallel", "k_lateral", "k_rotation", "d_parallel", "d_lateral", "d_rotation", "internal_force"))
        },
        "jam": int(any_jam),
        "final_stage": env.stage.name,
        "failure_reason": failure_reason(env, any_jam),
    }
    return row, trace


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    p = successes / total
    denominator = 1.0 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return center - half, center + half


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = sum(int(row["success"]) for row in rows)
    low, high = wilson_interval(successes, len(rows))
    numeric = (
        "mean_depth_m", "min_depth_m", "mean_lateral_error_m", "max_lateral_error_m",
        "orientation_error_deg", "peak_contact_force_N", "mean_force_all_steps_N",
        "mean_force_contact_steps_N", "completion_time_s", "steps", "jam",
        "mean_trajectory_residual_norm", "peak_trajectory_residual_norm",
        "mean_impedance_residual_norm", "peak_impedance_residual_norm",
        "mean_impedance_k_parallel", "mean_impedance_k_lateral", "mean_impedance_k_rotation",
        "mean_impedance_d_parallel", "mean_impedance_d_lateral", "mean_impedance_d_rotation",
        "mean_impedance_internal_force",
    )
    summary: dict[str, Any] = {
        "episodes": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows) if rows else float("nan"),
        "success_ci95_low": low,
        "success_ci95_high": high,
        "mean_success_completion_time_s": (
            float(np.mean([float(row["completion_time_s"]) for row in rows if int(row["success"]) == 1]))
            if successes else None
        ),
    }
    for key in numeric:
        values = np.asarray([float(row[key]) for row in rows], dtype=float)
        summary[f"mean_{key}"] = float(np.mean(values)) if len(values) else float("nan")
        summary[f"median_{key}"] = float(np.median(values)) if len(values) else float("nan")
    reasons: dict[str, int] = {}
    for row in rows:
        reason = str(row["failure_reason"])
        reasons[reason] = reasons.get(reason, 0) + 1
    summary["failure_reason_distribution"] = reasons
    return summary


def validation_score(summary: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(summary["success_rate"]),
        -float(summary["mean_jam"]),
        -float(summary["mean_peak_contact_force_N"]),
        -float(summary["mean_steps"]),
    )


def training_episode_row(trainer: A2POTrainer, record, trace_start: int, curriculum_name: str) -> dict[str, Any]:
    trace = trainer.impedance_trace[trace_start:]
    lateral, depth = trainer.env.peg_errors()
    forces = [float(np.linalg.norm([row["Fx"], row["Fy"], row["Fz"]])) for row in trace]
    return {
        **asdict(record),
        "curriculum": curriculum_name,
        "mean_depth_m": float(np.mean(depth)),
        "min_depth_m": float(np.min(depth)),
        "mean_lateral_error_m": float(np.mean(lateral)),
        "max_lateral_error_m": float(np.max(lateral)),
        "orientation_error_deg": float(trainer.env._relative_orientation_error()),
        "mean_force_all_steps_N": float(np.mean(forces)) if forces else 0.0,
        "failure_reason": failure_reason(trainer.env, record.jamming),
    }


def train_seed(args: argparse.Namespace) -> int:
    torch.set_num_threads(1)
    base = load_base_config()
    protocol = protocol_payload(base)
    seed_dir = Path(args.output_root) / f"seed_{args.seed}"
    checkpoint_dir = seed_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / "protocol.json").write_text(json.dumps(protocol, indent=2))
    train_cfg = A2POTrainConfig(
        seed=args.seed,
        formal_episodes=args.episodes,
        checkpoint_every=args.validate_every,
        device="cpu",
    )
    initial_env = DynamicVerticalDualPandaEnv(curriculum(base, 0, args.episodes), seed=args.seed)
    trainer = A2POTrainer(initial_env, train_cfg, seed_dir)
    if args.initial_checkpoint:
        load_weights(trainer, Path(args.initial_checkpoint))
    training_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    best_score: tuple[float, float, float, float] | None = None
    best_episode = 0
    for episode in range(args.episodes):
        cfg = curriculum(base, episode, args.episodes)
        fraction = episode / max(1, args.episodes - 1)
        curriculum_name = "easy" if fraction < 1/3 else "medium" if fraction < 2/3 else "hard"
        trainer.env = DynamicVerticalDualPandaEnv(cfg, seed=args.seed + episode)
        trace_start = len(trainer.impedance_trace)
        record = trainer.train_episode(episode)
        trainer.records.append(record)
        training_rows.append(training_episode_row(trainer, record, trace_start, curriculum_name))
        if (episode + 1) % args.validate_every == 0 or episode + 1 == args.episodes:
            checkpoint = checkpoint_dir / f"checkpoint_{episode + 1:06d}.pt"
            trainer.save_checkpoint(episode + 1, checkpoint)
            rows = []
            val_cfg = evaluation_config(base, "hard")
            for val_seed in VALIDATION_SEEDS[:args.validation_episodes]:
                row, _ = rollout_episode(trainer, val_cfg, val_seed, "a2po")
                rows.append(row)
            summary = summarize_rows(rows)
            validation_rows.append({"episode": episode + 1, **{k: v for k, v in summary.items() if not isinstance(v, dict)}})
            score = validation_score(summary)
            if best_score is None or score > best_score:
                best_score = score
                best_episode = episode + 1
                shutil.copy2(checkpoint, seed_dir / "checkpoint_best.pt")
            print(json.dumps({"seed": args.seed, "episode": episode + 1, "validation": summary, "best_episode": best_episode}), flush=True)
    write_csv(seed_dir / "training_episodes.csv", training_rows)
    write_csv(seed_dir / "validation_history.csv", validation_rows)
    selection = {
        "train_seed": args.seed,
        "episodes": args.episodes,
        "warm_start_checkpoint": str(args.initial_checkpoint) if args.initial_checkpoint else None,
        "best_episode": best_episode,
        "best_score": list(best_score or ()),
        "best_checkpoint": str(seed_dir / "checkpoint_best.pt"),
        "validation_episodes": args.validation_episodes,
        "validation_seeds": list(VALIDATION_SEEDS[:args.validation_episodes]),
    }
    (seed_dir / "checkpoint_selection.json").write_text(json.dumps(selection, indent=2))
    return 0


def evaluate(args: argparse.Namespace) -> int:
    torch.set_num_threads(1)
    base = load_base_config()
    cfg = evaluation_config(base, args.setting)
    train_cfg = A2POTrainConfig(seed=args.train_seed, device="cpu")
    trainer = A2POTrainer(DynamicVerticalDualPandaEnv(cfg, seed=0), train_cfg, Path(args.output).parent)
    if args.variant != "controller_only":
        load_weights(trainer, Path(args.checkpoint))
        trainer.agent1.eval(); trainer.agent2.eval()
    rows = []
    seed_start = EVALUATION_SEED_START[args.setting]
    for index in range(args.episodes):
        row, _ = rollout_episode(trainer, cfg, seed_start + index, args.variant)
        row.update({"setting": args.setting, "train_seed": args.train_seed, "checkpoint": str(args.checkpoint)})
        rows.append(row)
        if (index + 1) % 50 == 0:
            print(json.dumps({"setting": args.setting, "variant": args.variant, "train_seed": args.train_seed, "episodes": index + 1}), flush=True)
    output = Path(args.output)
    write_csv(output, rows)
    output.with_suffix(".summary.json").write_text(json.dumps(summarize_rows(rows), indent=2))
    return 0


def numeric_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    integer_keys = {"seed", "train_seed", "success", "steps", "contact_steps", "jam"}
    numeric_keys = {
        "completion_time_s", "peg1_depth_m", "peg2_depth_m", "mean_depth_m", "min_depth_m",
        "peg1_lateral_error_m", "peg2_lateral_error_m", "mean_lateral_error_m",
        "max_lateral_error_m", "orientation_error_deg", "peak_contact_force_N",
        "mean_force_all_steps_N", "mean_force_contact_steps_N", "peak_contact_torque_Nm",
        "mean_trajectory_residual_norm", "peak_trajectory_residual_norm",
        "mean_impedance_residual_norm", "peak_impedance_residual_norm",
        "mean_impedance_k_parallel", "mean_impedance_k_lateral", "mean_impedance_k_rotation",
        "mean_impedance_d_parallel", "mean_impedance_d_lateral", "mean_impedance_d_rotation",
        "mean_impedance_internal_force",
    }
    converted = []
    for row in rows:
        out: dict[str, Any] = dict(row)
        for key in integer_keys:
            if key in out:
                out[key] = int(out[key])
        for key in numeric_keys:
            if key in out:
                out[key] = float(out[key])
        converted.append(out)
    return converted


def paired_comparison(a2po_rows: list[dict[str, Any]], controller_rows: list[dict[str, Any]]) -> dict[str, Any]:
    a = {int(row["seed"]): row for row in a2po_rows}
    c = {int(row["seed"]): row for row in controller_rows}
    seeds = sorted(set(a) & set(c))
    rl_only = sum(int(a[s]["success"] == 1 and c[s]["success"] == 0) for s in seeds)
    controller_only = sum(int(a[s]["success"] == 0 and c[s]["success"] == 1) for s in seeds)
    discordant = rl_only + controller_only
    p_value = float(binomtest(min(rl_only, controller_only), discordant, 0.5).pvalue) if discordant else 1.0
    success_differences = np.asarray([a[s]["success"] - c[s]["success"] for s in seeds], dtype=float)
    force_differences = np.asarray([a[s]["peak_contact_force_N"] - c[s]["peak_contact_force_N"] for s in seeds], dtype=float)
    rng = np.random.default_rng(20260825)
    if seeds:
        indices = rng.integers(0, len(seeds), size=(10000, len(seeds)))
        success_boot = success_differences[indices].mean(axis=1)
        force_boot = force_differences[indices].mean(axis=1)
        success_ci = np.quantile(success_boot, [0.025, 0.975]).tolist()
        force_ci = np.quantile(force_boot, [0.025, 0.975]).tolist()
    else:
        success_ci = [float("nan"), float("nan")]
        force_ci = [float("nan"), float("nan")]
    return {
        "paired_episodes": len(seeds),
        "a2po_only_successes": rl_only,
        "controller_only_successes": controller_only,
        "mcnemar_exact_two_sided_p": p_value,
        "success_rate_difference": float(success_differences.mean()) if len(seeds) else float("nan"),
        "success_difference_bootstrap_ci95": success_ci,
        "mean_peak_force_difference_N": float(force_differences.mean()) if len(seeds) else float("nan"),
        "peak_force_difference_bootstrap_ci95_N": force_ci,
    }


def two_way_success_bootstrap(rows: list[dict[str, Any]]) -> tuple[float, float]:
    train_seeds = sorted({int(row["train_seed"]) for row in rows})
    evaluation_seeds = sorted({int(row["seed"]) for row in rows})
    lookup = {(int(row["train_seed"]), int(row["seed"])): int(row["success"]) for row in rows}
    matrix = np.asarray(
        [[lookup[(train_seed, evaluation_seed)] for evaluation_seed in evaluation_seeds] for train_seed in train_seeds],
        dtype=float,
    )
    rng = np.random.default_rng(20260825)
    row_indices = rng.integers(0, len(train_seeds), size=(10000, len(train_seeds)))
    column_indices = rng.integers(0, len(evaluation_seeds), size=(10000, len(evaluation_seeds)))
    samples = matrix[row_indices[:, :, None], column_indices[:, None, :]].mean(axis=(1, 2))
    return tuple(float(value) for value in np.quantile(samples, [0.025, 0.975]))


def plot_reports(training_by_seed, validation_by_seed, ablation_rows, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for seed, rows in training_by_seed.items():
        episodes = np.asarray([int(row["episode"]) for row in rows])
        success = np.asarray([float(row["success"] in (True, "True", "1", 1)) for row in rows])
        reward = np.asarray([float(row["reward_total"]) for row in rows])
        window = 50
        rolling_success = np.convolve(success, np.ones(window)/window, mode="valid") if len(success) >= window else success
        rolling_reward = np.convolve(reward, np.ones(window)/window, mode="valid") if len(reward) >= window else reward
        x = episodes[window-1:] if len(success) >= window else episodes
        axes[0].plot(x, rolling_success, label=str(seed))
        axes[1].plot(x, rolling_reward, label=str(seed))
    axes[0].set(xlabel="training episode", ylabel="rolling success rate", ylim=(-.02, 1.02))
    axes[1].set(xlabel="training episode", ylabel="rolling total reward")
    for ax in axes: ax.grid(alpha=.25); ax.legend(title="seed")
    fig.tight_layout(); fig.savefig(figure_dir / "training_curves.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for seed, rows in validation_by_seed.items():
        ax.plot([int(row["episode"]) for row in rows], [float(row["success_rate"]) for row in rows], marker="o", label=str(seed))
    ax.set(xlabel="checkpoint episode", ylabel="validation success rate", ylim=(-.02, 1.02)); ax.grid(alpha=.25); ax.legend(title="seed")
    fig.tight_layout(); fig.savefig(figure_dir / "validation_convergence.png", dpi=180); plt.close(fig)

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in ablation_rows:
        groups.setdefault((row["setting"], row["variant"]), []).append(row)
    settings = ("nominal", "hard", "ood")
    variants = VARIANTS
    x = np.arange(len(settings)); width = .19
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, variant in enumerate(variants):
        variant_groups = [groups.get((setting, variant), []) for setting in settings]
        rates = [np.mean([r["success"] for r in rows]) for rows in variant_groups]
        intervals = [wilson_interval(sum(r["success"] for r in rows), len(rows)) for rows in variant_groups]
        errors = np.asarray([
            [max(0.0, rate-low) for rate, (low, _) in zip(rates, intervals)],
            [max(0.0, high-rate) for rate, (_, high) in zip(rates, intervals)],
        ])
        ax.bar(x + (i-1.5)*width, rates, width, yerr=errors, capsize=3, label=variant)
    ax.set_xticks(x, settings); ax.set_ylabel("success rate"); ax.set_ylim(0, 1.02); ax.grid(axis="y", alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(figure_dir / "success_rate_comparison.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [3, 1]})
    primary = variants[:3]
    primary_width = .24
    for i, variant in enumerate(primary):
        forces = [np.mean([r["peak_contact_force_N"] for r in groups.get((setting, variant), [])]) for setting in settings]
        axes[0].bar(x + (i-1)*primary_width, forces, primary_width, label=variant)
    axes[0].set_xticks(x, settings); axes[0].set_ylabel("mean episode peak contact force (N)"); axes[0].set_title("primary methods")
    axes[0].grid(axis="y", alpha=.25); axes[0].legend()
    no_stage_forces = [np.mean([r["peak_contact_force_N"] for r in groups.get((setting, variants[3]), [])]) for setting in settings]
    axes[1].bar(settings, no_stage_forces, color="tab:red")
    axes[1].set_title("no staged alignment/recovery"); axes[1].grid(axis="y", alpha=.25)
    axes[1].tick_params(axis="x", rotation=18)
    fig.tight_layout(); fig.savefig(figure_dir / "force_comparison.png", dpi=180); plt.close(fig)

    reasons = sorted({row["failure_reason"] for row in ablation_rows if row["failure_reason"] != "success"})
    hard = [row for row in ablation_rows if row["setting"] == "hard"]
    fig, ax = plt.subplots(figsize=(12, 6)); bottom = np.zeros(len(variants))
    for reason in reasons:
        counts = np.asarray([sum(row["failure_reason"] == reason for row in hard if row["variant"] == variant) for variant in variants])
        ax.bar(variants, counts, bottom=bottom, label=reason); bottom += counts
    ax.set_ylabel("hard-evaluation episodes"); ax.tick_params(axis="x", rotation=18); ax.legend(); ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(figure_dir / "failure_mode_comparison.png", dpi=180); plt.close(fig)


def report(args: argparse.Namespace) -> int:
    base = load_base_config()
    regression = regression_result()
    seed_dirs = [FORMAL_ROOT / f"seed_{seed}" for seed in TRAIN_SEEDS]
    training_by_seed = {seed: read_csv(path / "training_episodes.csv") for seed, path in zip(TRAIN_SEEDS, seed_dirs)}
    validation_by_seed = {seed: read_csv(path / "validation_history.csv") for seed, path in zip(TRAIN_SEEDS, seed_dirs)}
    selections = [json.loads((path / "checkpoint_selection.json").read_text()) for path in seed_dirs]
    best_selection = max(selections, key=lambda item: tuple(item["best_score"]))
    best_seed = int(best_selection["train_seed"])

    final_rows: list[dict[str, Any]] = []
    for seed in TRAIN_SEEDS:
        for setting in EVALUATION_SEED_START:
            path = FORMAL_ROOT / "evaluation" / f"seed_{seed}_a2po_{setting}.csv"
            final_rows.extend(numeric_rows(read_csv(path)))
    write_csv(RESULT_ROOT / "final_evaluation.csv", final_rows)

    ablation_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for setting in EVALUATION_SEED_START:
            path = FORMAL_ROOT / "ablation" / f"seed_{best_seed}_{variant}_{setting}.csv"
            ablation_rows.extend(numeric_rows(read_csv(path)))
    write_csv(RESULT_ROOT / "ablation.csv", ablation_rows)

    per_seed_rows = []
    per_seed_summaries: dict[str, Any] = {}
    for seed in TRAIN_SEEDS:
        per_seed_summaries[str(seed)] = {}
        for setting in EVALUATION_SEED_START:
            rows = [row for row in final_rows if row["train_seed"] == seed and row["setting"] == setting]
            summary = summarize_rows(rows)
            per_seed_summaries[str(seed)][setting] = summary
            per_seed_rows.append({"train_seed": seed, "setting": setting, **{k: v for k, v in summary.items() if not isinstance(v, dict)}})
    write_csv(RESULT_ROOT / "per_seed_results.csv", per_seed_rows)
    aggregate_a2po = {
        setting: summarize_rows([row for row in final_rows if row["setting"] == setting])
        for setting in EVALUATION_SEED_START
    }
    for setting in EVALUATION_SEED_START:
        setting_rows = [row for row in final_rows if row["setting"] == setting]
        low, high = two_way_success_bootstrap(setting_rows)
        aggregate_a2po[setting]["multi_seed_two_way_bootstrap_ci95_low"] = low
        aggregate_a2po[setting]["multi_seed_two_way_bootstrap_ci95_high"] = high
        aggregate_a2po[setting]["multi_seed_ci_method"] = "two-way bootstrap over training and shared evaluation seeds"

    ablation_summaries: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    for setting in EVALUATION_SEED_START:
        ablation_summaries[setting] = {}
        for variant in VARIANTS:
            rows = [row for row in ablation_rows if row["setting"] == setting and row["variant"] == variant]
            ablation_summaries[setting][variant] = summarize_rows(rows)
        a2po_rows = [row for row in ablation_rows if row["setting"] == setting and row["variant"] == "a2po"]
        controller_rows = [row for row in ablation_rows if row["setting"] == setting and row["variant"] == "controller_only"]
        paired[setting] = paired_comparison(a2po_rows, controller_rows)

    validation_tail = []
    validation_initial_rates = []
    validation_tail_rates = []
    for seed, rows in validation_by_seed.items():
        rates = np.asarray([float(row["success_rate"]) for row in rows], dtype=float)
        tail = rates[-3:]
        initial = rates[:2]
        validation_initial_rates.extend(initial.tolist())
        validation_tail_rates.extend(tail.tolist())
        validation_tail.append({"seed": seed, "last_three": tail.tolist(), "std": float(np.std(tail)), "slope": float(np.polyfit(np.arange(len(tail)), tail, 1)[0]) if len(tail) > 1 else 0.0})
    hard_rates = np.asarray([per_seed_summaries[str(seed)]["hard"]["success_rate"] for seed in TRAIN_SEEDS])
    ood_rates = np.asarray([per_seed_summaries[str(seed)]["ood"]["success_rate"] for seed in TRAIN_SEEDS])
    multi_seed_stable = bool(np.std(hard_rates, ddof=1) <= 0.08 and np.std(ood_rates, ddof=1) <= 0.08)
    validation_plateau_stable = bool(sum(item["std"] <= 0.10 and abs(item["slope"]) <= 0.05 for item in validation_tail) >= 4)
    validation_improvement = float(np.mean(validation_tail_rates) - np.mean(validation_initial_rates))
    convergence = {
        "validation_tail": validation_tail,
        "validation_initial_mean": float(np.mean(validation_initial_rates)),
        "validation_tail_mean": float(np.mean(validation_tail_rates)),
        "validation_improvement_absolute": validation_improvement,
        "hard_success_mean_across_seeds": float(np.mean(hard_rates)),
        "hard_success_std_across_seeds": float(np.std(hard_rates, ddof=1)),
        "ood_success_mean_across_seeds": float(np.mean(ood_rates)),
        "ood_success_std_across_seeds": float(np.std(ood_rates, ddof=1)),
        "multi_seed_performance_stable": multi_seed_stable,
        "validation_plateau_stable": validation_plateau_stable,
        "converged_to_improved_policy": bool(validation_plateau_stable and validation_improvement >= 0.05),
    }
    hard_pair = paired["hard"]
    ood_pair = paired["ood"]
    claim_supported = bool(
        convergence["converged_to_improved_policy"]
        and multi_seed_stable
        and hard_pair["mcnemar_exact_two_sided_p"] < 0.05
        and hard_pair["success_rate_difference"] > 0
        and ood_pair["success_rate_difference"] > 0
        and hard_pair["peak_force_difference_bootstrap_ci95_N"][1] <= 5.0
        and ood_pair["peak_force_difference_bootstrap_ci95_N"][1] <= 5.0
        and regression["all_passed"]
    )
    summary = {
        "protocol": protocol_payload(base),
        "frozen_artifact_sha256": frozen_artifact_hashes(),
        "regression_tests": regression,
        "checkpoint_selections": selections,
        "best_seed_for_ablation_selected_on_validation_only": best_seed,
        "per_seed": per_seed_summaries,
        "aggregate_a2po_all_seeds": aggregate_a2po,
        "ablation": ablation_summaries,
        "paired_a2po_vs_controller": paired,
        "convergence": convergence,
        "acceptance": {
            "a2po_converged": convergence["converged_to_improved_policy"],
            "validation_plateau_stable": convergence["validation_plateau_stable"],
            "multi_seed_performance_stable": multi_seed_stable,
            "hard_above_previous_40_percent": bool(aggregate_a2po["hard"]["multi_seed_two_way_bootstrap_ci95_low"] > 0.40),
            "regression_tests_23_of_23_pass": regression["all_passed"],
            "rl_improves_controller_claim_supported": claim_supported,
        },
        "leakage_audit": {
            "validation_and_test_seeds_disjoint": True,
            "test_metrics_used_for_checkpoint_selection": False,
            "success_or_failure_label_in_policy_input": False,
            "future_state_in_policy_input": False,
            "deterministic_prior_shared_by_controller_and_a2po": True,
            "hard_and_ood_initial_seeds_shared_across_methods": True,
        },
    }
    (RESULT_ROOT / "final_summary.json").write_text(json.dumps(summary, indent=2))
    plot_reports(training_by_seed, validation_by_seed, ablation_rows, FIGURE_ROOT)
    print(json.dumps(summary["acceptance"], indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train-seed")
    train.add_argument("--seed", type=int, required=True, choices=TRAIN_SEEDS)
    train.add_argument("--episodes", type=int, default=500)
    train.add_argument("--validate-every", type=int, default=50)
    train.add_argument("--validation-episodes", type=int, default=20)
    train.add_argument("--output-root", type=Path, default=FORMAL_ROOT)
    train.add_argument("--initial-checkpoint", type=Path, default=None, help="optional 7D Agent 2 checkpoint; legacy 6D checkpoints are incompatible")
    train.set_defaults(func=train_seed)

    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument("--train-seed", type=int, required=True)
    evaluate_parser.add_argument("--variant", choices=VARIANTS, required=True)
    evaluate_parser.add_argument("--setting", choices=tuple(EVALUATION_SEED_START), required=True)
    evaluate_parser.add_argument("--episodes", type=int, default=200)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.set_defaults(func=evaluate)

    report_parser = sub.add_parser("report")
    report_parser.set_defaults(func=report)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    FORMAL_ROOT.mkdir(parents=True, exist_ok=True)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
