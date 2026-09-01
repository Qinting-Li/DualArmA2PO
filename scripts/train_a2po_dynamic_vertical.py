#!/usr/bin/env python3
"""Train and evaluate A2PO on the cooperative dynamic-vertical task."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from x_bimanual.a2po_training import A2POTrainConfig, A2POTrainer  # noqa: E402
from x_bimanual.panda_dual_assembly import dual_config_from_mapping  # noqa: E402
from x_bimanual.panda_dynamic_vertical_assembly import DynamicVerticalDualPandaEnv  # noqa: E402


def curriculum(base, episode: int, total: int):
    fraction = episode / max(1, total - 1)
    if fraction < 1 / 3:
        return replace(base, initial_xy_range_m=0.002, initial_z_min_m=0.275, initial_z_max_m=0.285, initial_rotation_range_rad=0.008)
    if fraction < 2 / 3:
        return replace(base, initial_xy_range_m=0.004, initial_z_min_m=0.260, initial_z_max_m=0.300, initial_rotation_range_rad=0.020)
    return replace(base, initial_xy_range_m=0.006, initial_z_min_m=0.240, initial_z_max_m=0.320, initial_rotation_range_rad=0.040)


def write_records(path: Path, records) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0])))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def evaluate(trainer: A2POTrainer, cfg, episodes: int, seed: int):
    records = []
    for episode in range(episodes):
        trainer.env = DynamicVerticalDualPandaEnv(cfg, seed=seed + episode)
        records.append(trainer.run_episode(seed + episode, deterministic=True, collect_training=False)[0])
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/task.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "results/a2po_dual_panda_dynamic_vertical")
    parser.add_argument("--sanity-episodes", type=int, default=10)
    parser.add_argument("--train-episodes", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    with args.config.open() as handle:
        base = dual_config_from_mapping(yaml.safe_load(handle))
    base = replace(base, gravity=(0.0, 0.0, 0.0))
    if args.max_steps is not None:
        base = replace(base, max_steps=args.max_steps)
    train_cfg = A2POTrainConfig(seed=args.seed, sanity_episodes=args.sanity_episodes, formal_episodes=args.train_episodes, device=args.device)
    trainer = A2POTrainer(DynamicVerticalDualPandaEnv(curriculum(base, 0, max(1, args.train_episodes)), seed=args.seed), train_cfg, args.output)
    (args.output / "run_config.json").write_text(json.dumps({"train": asdict(train_cfg), "base_environment": asdict(base), "impedance_action_dim": trainer.impedance_action_dim}, indent=2))

    sanity = evaluate(trainer, curriculum(base, 0, 3), args.sanity_episodes, args.seed)
    sanity_summary = {"episodes": len(sanity), **trainer.summarize(sanity)}
    write_records(args.output / "sanity_episodes.csv", sanity)
    (args.output / "sanity_summary.json").write_text(json.dumps(sanity_summary, indent=2))
    print(json.dumps({"phase": "sanity", **sanity_summary}))

    trainer.records = []
    for episode in range(args.train_episodes):
        trainer.env = DynamicVerticalDualPandaEnv(curriculum(base, episode, args.train_episodes), seed=args.seed + episode)
        trainer.records.append(trainer.train_episode(episode))
        if (episode + 1) % trainer.config.checkpoint_every == 0:
            trainer.save_checkpoint(episode + 1)
        if (episode + 1) % 10 == 0:
            print(json.dumps({"phase": "training", "episode": episode + 1, **trainer.summarize(trainer.records[-10:])}))
    checkpoint = trainer.save_checkpoint(args.train_episodes, args.output / "checkpoint_final.pt")
    write_records(args.output / "training_episodes.csv", trainer.records)
    trainer.write_impedance_trace(args.output / "impedance_trace.csv")
    training_summary = {"episodes": len(trainer.records), **trainer.summarize(trainer.records)}
    (args.output / "training_summary.json").write_text(json.dumps(training_summary, indent=2))

    eval_cfg = curriculum(base, 2, 3)
    evaluation = evaluate(trainer, eval_cfg, args.eval_episodes, args.seed + 100000)
    evaluation_summary = {"episodes": len(evaluation), **trainer.summarize(evaluation)}
    write_records(args.output / "evaluation_episodes.csv", evaluation)
    (args.output / "evaluation_summary.json").write_text(json.dumps(evaluation_summary, indent=2))
    status = {"trained": True, "checkpoint": str(checkpoint), "environment": "DynamicVerticalDualPandaEnv", "impedance_action_dim": trainer.impedance_action_dim, "training": training_summary, "evaluation": evaluation_summary}
    (args.output / "dynamic_vertical_training_status.json").write_text(json.dumps(status, indent=2))
    demo_status = ROOT / "outputs/a2po_dual_panda_dynamic_vertical/dynamic_vertical_training_status.json"
    demo_status.parent.mkdir(parents=True, exist_ok=True)
    demo_status.write_text(json.dumps(status, indent=2))
    print(json.dumps({"phase": "complete", **status}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
