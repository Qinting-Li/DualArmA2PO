#!/usr/bin/env python3
"""Run A2PO sanity episodes followed by formal two-policy PPO training."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from x_bimanual.a2po_training import A2POTrainConfig, A2POTrainer  # noqa: E402
from x_bimanual.panda_dual_assembly import DualPandaAssemblyEnv, dual_config_from_mapping  # noqa: E402


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def write_records(path: Path, records) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(records[0]).keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def make_plots(records, output_dir: Path) -> None:
    if not records:
        return
    episodes = np.asarray([r.episode for r in records])
    r1 = np.asarray([r.reward_agent1 for r in records])
    r2 = np.asarray([r.reward_agent2 for r in records])
    rt = np.asarray([r.reward_total for r in records])
    success = np.asarray([r.success for r in records], dtype=float)
    force = np.asarray([r.peak_force_N for r in records])
    jam = np.asarray([r.jamming for r in records], dtype=float)

    def rolling(values: np.ndarray, window: int = 25) -> np.ndarray:
        result = np.empty_like(values, dtype=float)
        for i in range(len(values)):
            result[i] = values[max(0, i - window + 1): i + 1].mean()
        return result

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, r1, alpha=0.2, label="Agent 1 reward")
    ax.plot(episodes, r2, alpha=0.2, label="Agent 2 reward")
    ax.plot(episodes, rolling(rt), label="cooperative reward rolling")
    ax.set_xlabel("episode"); ax.set_ylabel("reward"); ax.grid(alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(output_dir / "learning_curve.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(episodes, rolling(success), label="success rate")
    ax[0].set_ylim(-0.02, 1.02); ax[0].set_ylabel("success"); ax[0].legend()
    ax[1].plot(episodes, force, label="peak contact force (N)")
    ax[1].set_ylabel("N"); ax[1].legend()
    ax[2].plot(episodes, rolling(jam), label="jamming rate")
    ax[2].set_xlabel("episode"); ax[2].set_ylabel("jam"); ax[2].legend()
    for axis in ax: axis.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(output_dir / "success_force_jamming.png", dpi=150); plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/task.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "results/a2po_dual_panda")
    parser.add_argument("--sanity-episodes", type=int, default=100)
    parser.add_argument("--formal-episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-formal", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    with args.config.open() as handle:
        mapping = yaml.safe_load(handle)
    env_config = dual_config_from_mapping(mapping)
    env = DualPandaAssemblyEnv(env_config, seed=args.seed)
    train_config = A2POTrainConfig(seed=args.seed, sanity_episodes=args.sanity_episodes, formal_episodes=args.formal_episodes, device=args.device)
    trainer = A2POTrainer(env, train_config, args.output)
    write_json(args.output / "run_config.json", {"train": asdict(train_config), "environment": asdict(env_config), "observation_shapes": env.observation_space_shapes})

    sanity_records = [trainer.run_episode(i, deterministic=True, collect_training=False)[0] for i in range(args.sanity_episodes)]
    sanity = trainer.summarize(sanity_records)
    sanity.update({"episodes": args.sanity_episodes, "action_range_valid": sanity["action1_min"] >= -1.00001 and sanity["action1_max"] <= 1.00001 and sanity["action2_min"] >= -1e-6 and sanity["action2_max"] <= 1.00001, "impedance_positive": sanity["kp_min"] > 0 and sanity["kd_min"] > 0, "finite_metrics": all(np.isfinite(float(value)) for value in sanity.values())})
    write_json(args.output / "sanity_summary.json", sanity)
    write_records(args.output / "sanity_episodes.csv", sanity_records)
    trainer.impedance_trace = [row for row in trainer.impedance_trace if row["episode"] < args.sanity_episodes]
    trainer.write_impedance_trace(args.output / "sanity_impedance_trace.csv")
    print(json.dumps({"phase": "sanity", **sanity}, indent=2))
    if not (sanity["action_range_valid"] and sanity["impedance_positive"] and sanity["finite_metrics"]):
        raise RuntimeError("sanity checks failed; formal training was not started")
    if args.skip_formal:
        return 0

    # Formal training starts only after the completed sanity run.  The same
    # env instance is reset between episodes; no physical model parameters or
    # geometry are changed by the trainer.
    trainer.impedance_trace.clear()
    for episode in range(args.formal_episodes):
        record = trainer.train_episode(episode)
        trainer.records.append(record)
        if (episode + 1) % 25 == 0:
            recent = trainer.summarize(trainer.records[-25:])
            print(json.dumps({"phase": "formal", "episode": episode + 1, **recent}))
        if (episode + 1) % train_config.checkpoint_every == 0:
            trainer.save_checkpoint(episode + 1)
    trainer.save_checkpoint(args.formal_episodes, args.output / "checkpoint_final.pt")
    trainer.write_records(args.output / "formal_episodes.csv")
    trainer.write_impedance_trace(args.output / "impedance_trace.csv")
    summary = trainer.summarize(trainer.records)
    summary["formal_episodes"] = args.formal_episodes
    write_json(args.output / "formal_summary.json", summary)
    make_plots(trainer.records, args.output)
    print(json.dumps({"phase": "formal_complete", **summary, "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
