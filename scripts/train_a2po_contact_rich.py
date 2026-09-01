#!/usr/bin/env python3
"""Contact-rich curriculum training, evaluation, and ablations for dual-Panda A2PO."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from x_bimanual.a2po_training import A2POTrainConfig, A2POTrainer, impedance_prior  # noqa: E402
from x_bimanual.panda_dual_assembly import DualPandaAssemblyEnv, dual_config_from_mapping  # noqa: E402


def curriculum(base, episode: int, total: int):
    fraction = episode / max(1, total - 1)
    if fraction < 1 / 3:
        return replace(base, initial_xy_range_m=0.004, initial_z_min_m=0.328, initial_z_max_m=0.332, initial_rotation_range_rad=0.02)
    if fraction < 2 / 3:
        return replace(base, initial_xy_range_m=0.006, initial_z_min_m=0.260, initial_z_max_m=0.300, initial_rotation_range_rad=0.04)
    # Difficult episodes begin at the receiver mouth with randomized lateral
    # and angular error, forcing the policies to experience physical contact.
    return replace(base, initial_xy_range_m=0.006, initial_z_min_m=0.225, initial_z_max_m=0.245, initial_rotation_range_rad=0.06)


def write_records(path: Path, records) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in records)


def rolling(values, window=50):
    values = np.asarray(values, dtype=float)
    return np.asarray([values[max(0, i - window + 1):i + 1].mean() for i in range(len(values))])


def save_training_plots(records, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    ep = np.asarray([r.episode for r in records])
    total = np.asarray([r.reward_total for r in records])
    success = np.asarray([r.success for r in records], dtype=float)
    force = np.asarray([r.peak_force_N for r in records])
    jam = np.asarray([r.jamming for r in records], dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5)); ax.plot(ep, total, alpha=.2); ax.plot(ep, rolling(total), label="cooperative reward"); ax.set(xlabel="episode", ylabel="reward"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(figure_dir / "learning_curve.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5)); ax.plot(ep, rolling(success), label="success rate"); ax.set_ylim(-.02, 1.02); ax.set(xlabel="episode", ylabel="rate"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(figure_dir / "success_rate_comparison.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5)); ax.plot(ep, force, alpha=.25); ax.plot(ep, rolling(force), label="peak contact force"); ax.set(xlabel="episode", ylabel="N"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(figure_dir / "contact_force_comparison.png", dpi=160); plt.close(fig)


def save_trace_plots(trace: list[dict], figure_dir: Path) -> None:
    if not trace:
        return
    figure_dir.mkdir(parents=True, exist_ok=True)
    stage = np.asarray([int(row.get("stage_index", 0)) for row in trace])
    x = np.arange(len(trace))
    stages = [(i, name) for i, name in enumerate(("INITIALIZATION", "GRASP", "LIFT", "TRANSPORT", "COARSE_ALIGNMENT", "APPROACH", "FIRST_CONTACT", "COMPLIANT_ALIGNMENT", "INSERTION", "SUCCESS"))]
    for filename, columns, ylabel in (("impedance_stiffness_vs_time.png", ("Kx", "Ky", "Kz", "Krx", "Kry", "Krz"), "stiffness"), ("impedance_damping_vs_time.png", ("Dx", "Dy", "Dz", "Drx", "Dry", "Drz"), "damping")):
        fig, ax = plt.subplots(figsize=(12, 5))
        for column in columns: ax.plot(x, [row[column] for row in trace], label=column, linewidth=.8)
        for index, name in stages:
            if np.any(stage == index): ax.axvline(int(np.flatnonzero(stage == index)[0]), color="k", alpha=.12)
        ax.set(xlabel="control step (stage boundaries marked)", ylabel=ylabel); ax.grid(alpha=.2); ax.legend(ncol=3); fig.tight_layout(); fig.savefig(figure_dir / filename, dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5)); ax.plot(x, [np.linalg.norm([row["Fx"], row["Fy"], row["Fz"]]) for row in trace], label="contact force N"); ax.plot(x, [row["peg1_depth"] + row["peg2_depth"] for row in trace], label="sum insertion depth m"); ax.set_xlabel("control step; stage boundaries marked"); ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(figure_dir / "force_and_insertion_depth.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5)); ax.plot(x, [row["peg1_lateral_error"] for row in trace], label="peg 1"); ax.plot(x, [row["peg2_lateral_error"] for row in trace], label="peg 2"); ax.set(xlabel="control step; stage boundaries marked", ylabel="lateral error m"); ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(figure_dir / "peg_alignment_errors.png", dpi=160); plt.close(fig)


def evaluate_variant(trainer: A2POTrainer, variant: str, episodes: int, cfg, seed: int) -> tuple[dict, list[dict]]:
    rows = []
    for episode in range(episodes):
        trainer.env = DualPandaAssemblyEnv(cfg, seed=seed + episode)
        obs = trainer.env.reset(seed + episode)
        forces = []; torques = []; jams = 0; steps = 0
        for step in range(trainer.env.cfg.max_steps):
            a1, _, _, _ = trainer.agent1.sample(obs["trajectory"], deterministic=True, prior_override=trainer._object_target_prior())
            if variant == "A_fixed_impedance":
                a2 = np.full(12, 0.5, dtype=np.float32)
            else:
                if variant == "B_unconditioned_agent2":
                    conditioning = np.zeros(6, dtype=np.float32)
                elif variant == "C_prior_condition":
                    conditioning = trainer.agent1.prior_fn(obs["trajectory"])
                else:
                    conditioning = a1
                a2, _, _, _ = trainer.agent2.sample(np.r_[obs["impedance"], conditioning], deterministic=True)
            obs, _, done, info = trainer.env.step(a1, a2)
            wrench = np.asarray(info.get("wrench", np.zeros(6)), dtype=float)
            forces.append(float(np.linalg.norm(wrench[:3]))); torques.append(float(np.linalg.norm(wrench[3:]))); jams += int(info.get("jamming", False)); steps += 1
            if done: break
        lat, dep = trainer.env.peg_errors()
        rows.append({"variant": variant, "episode": episode, "success": int(trainer.env.success), "peak_force_N": max(forces, default=0.0), "mean_contact_force_N": float(np.mean(forces)) if forces else 0.0, "peak_torque_Nm": max(torques, default=0.0), "insertion_time_steps": steps, "jamming": int(jams > 0), "jamming_steps": jams, "position_error_m": float(np.linalg.norm(trainer.env.desired_pose[:3] - trainer.env._workpiece_pose()[0])), "orientation_error_deg": trainer.env._relative_orientation_error(), "peg1_lateral_error_m": float(lat[0]), "peg2_lateral_error_m": float(lat[1]), "peg1_insertion_depth_m": float(dep[0]), "peg2_insertion_depth_m": float(dep[1])})
    keys = rows[0].keys() if rows else []
    summary = {"variant": variant, "episodes": episodes, "success_rate": float(np.mean([r["success"] for r in rows])) if rows else 0.0, "peak_contact_force_N": float(np.max([r["peak_force_N"] for r in rows])) if rows else 0.0, "mean_contact_force_N": float(np.mean([r["mean_contact_force_N"] for r in rows])) if rows else 0.0, "peak_torque_Nm": float(np.max([r["peak_torque_Nm"] for r in rows])) if rows else 0.0, "mean_insertion_time_steps": float(np.mean([r["insertion_time_steps"] for r in rows])) if rows else 0.0, "jamming_rate": float(np.mean([r["jamming"] for r in rows])) if rows else 0.0, "mean_position_error_m": float(np.mean([r["position_error_m"] for r in rows])) if rows else 0.0, "mean_orientation_error_deg": float(np.mean([r["orientation_error_deg"] for r in rows])) if rows else 0.0, "mean_peg1_lateral_error_m": float(np.mean([r["peg1_lateral_error_m"] for r in rows])) if rows else 0.0, "mean_peg2_lateral_error_m": float(np.mean([r["peg2_lateral_error_m"] for r in rows])) if rows else 0.0, "mean_peg1_insertion_depth_m": float(np.mean([r["peg1_insertion_depth_m"] for r in rows])) if rows else 0.0, "mean_peg2_insertion_depth_m": float(np.mean([r["peg2_insertion_depth_m"] for r in rows])) if rows else 0.0}
    return summary, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/task.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "results/a2po_dual_panda_final")
    parser.add_argument("--figures", type=Path, default=ROOT / "figures/a2po_dual_panda_final")
    parser.add_argument("--sanity-episodes", type=int, default=100)
    parser.add_argument("--train-episodes", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True); args.figures.mkdir(parents=True, exist_ok=True)
    with args.config.open() as handle: base = dual_config_from_mapping(yaml.safe_load(handle))
    train_cfg = A2POTrainConfig(seed=args.seed, sanity_episodes=args.sanity_episodes, formal_episodes=args.train_episodes, device=args.device)
    trainer = A2POTrainer(DualPandaAssemblyEnv(curriculum(base, 0, max(1, args.train_episodes)), seed=args.seed), train_cfg, args.output)
    (args.output / "run_config.json").write_text(json.dumps({"train": asdict(train_cfg), "base_environment": asdict(base), "curriculum": {"easy": asdict(curriculum(base, 0, 3)), "medium": asdict(curriculum(base, 1, 3)), "difficult": asdict(curriculum(base, 2, 3))}}, indent=2))
    sanity_records = []
    for episode in range(args.sanity_episodes):
        trainer.env = DualPandaAssemblyEnv(curriculum(base, episode, max(1, args.sanity_episodes)), seed=args.seed + episode)
        sanity_records.append(trainer.run_episode(episode, deterministic=True, collect_training=False)[0])
    sanity = trainer.summarize(sanity_records); sanity.update({"episodes": args.sanity_episodes, "action_range_valid": sanity["action1_min"] >= -1.00001 and sanity["action1_max"] <= 1.00001 and sanity["action2_min"] >= -1e-6 and sanity["action2_max"] <= 1.00001, "impedance_positive": sanity["kp_min"] > 0 and sanity["kd_min"] > 0})
    (args.output / "sanity_summary.json").write_text(json.dumps(sanity, indent=2)); write_records(args.output / "sanity_episodes.csv", sanity_records)
    print(json.dumps({"phase": "sanity", **sanity}, indent=2))
    if not sanity["action_range_valid"] or not sanity["impedance_positive"]: raise RuntimeError("sanity checks failed")
    trainer.records = []
    for episode in range(args.train_episodes):
        trainer.env = DualPandaAssemblyEnv(curriculum(base, episode, args.train_episodes), seed=args.seed + episode)
        record = trainer.train_episode(episode); trainer.records.append(record)
        if (episode + 1) % train_cfg.checkpoint_every == 0: trainer.save_checkpoint(episode + 1)
        if (episode + 1) % 50 == 0: print(json.dumps({"phase": "training", "episode": episode + 1, **trainer.summarize(trainer.records[-50:])}))
    trainer.save_checkpoint(args.train_episodes, args.output / "checkpoint_final.pt"); write_records(args.output / "training_episodes.csv", trainer.records); trainer.write_impedance_trace(args.output / "impedance_trace.csv")
    (args.output / "training_summary.json").write_text(json.dumps({"episodes": args.train_episodes, **trainer.summarize(trainer.records)}, indent=2)); save_training_plots(trainer.records, args.figures); save_trace_plots(trainer.impedance_trace, args.figures)
    summaries = []; all_rows = []
    eval_cfg = replace(base, initial_xy_range_m=0.006, initial_z_min_m=0.225, initial_z_max_m=0.245, initial_rotation_range_rad=0.06)
    for variant in ("A_fixed_impedance", "B_unconditioned_agent2", "C_prior_condition", "D_full_A2PO"):
        summary, rows = evaluate_variant(trainer, variant, args.eval_episodes, eval_cfg, args.seed + 100000)
        summaries.append(summary); all_rows.extend(rows); print(json.dumps({"phase": "evaluation", **summary}, indent=2))
    with (args.output / "ablation_episodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)
    (args.output / "formal_evaluation_summary.json").write_text(json.dumps({"episodes_per_variant": args.eval_episodes, "variants": summaries}, indent=2))
    fig, axes = plt.subplots(1, 3, figsize=(14, 5)); names = [s["variant"] for s in summaries]; x = np.arange(len(names)); axes[0].bar(x, [s["success_rate"] for s in summaries]); axes[0].set_ylabel("success rate"); axes[1].bar(x, [s["mean_contact_force_N"] for s in summaries]); axes[1].set_ylabel("mean contact force (N)"); axes[2].bar(x, [s["jamming_rate"] for s in summaries]); axes[2].set_ylabel("jamming rate");
    for ax in axes: ax.set_xticks(x, names, rotation=25, ha="right"); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(args.figures / "ablation_comparison.png", dpi=160); plt.close(fig)
    # A second comparison plot is intentionally derived from the same formal
    # evaluation, making the fixed-impedance baseline directly auditable.
    fig, ax = plt.subplots(figsize=(10, 5)); ax.bar(names, [s["success_rate"] for s in summaries], color=["#777", "#4c78a8", "#f58518", "#54a24b"]); ax.set_ylim(0, 1.05); ax.set_ylabel("success rate"); ax.tick_params(axis="x", rotation=25); ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(args.figures / "success_rate_comparison.png", dpi=160); plt.close(fig)
    print(json.dumps({"phase": "complete", "output": str(args.output), "figures": str(args.figures)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
