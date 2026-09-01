#!/usr/bin/env python3
"""Train A2PO on tabletop pickup followed by contact-rich insertion."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import yaml
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from x_bimanual.a2po_training import A2POTrainConfig, A2POTrainer  # noqa: E402
from x_bimanual.panda_dual_assembly import dual_config_from_mapping  # noqa: E402
from x_bimanual.panda_table_pickup_assembly import TablePickupDualPandaEnv  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/task.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "results/a2po_table_pickup")
    parser.add_argument("--train-episodes", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warm-start", type=Path, default=None, help="optional compatible A2PO checkpoint")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with args.config.open() as handle:
        cfg = dual_config_from_mapping(yaml.safe_load(handle))
    cfg = replace(
        cfg,
        gravity=(0.0, 0.0, -9.81),
        initial_xy_range_m=0.006,
        initial_z_min_m=0.287,
        initial_z_max_m=0.292,
        initial_rotation_range_rad=0.15,
        # Rubberized gripper contact for a physically stable tabletop lift.
        grasp_friction=2.0,
        internal_force_max=80.0,
        max_steps=args.max_steps,
    )
    train_cfg = A2POTrainConfig(
        seed=args.seed, sanity_episodes=0, formal_episodes=args.train_episodes,
        device=args.device, residual_scale_trajectory=0.12,
        residual_scale_impedance=0.08,
    )
    trainer = A2POTrainer(TablePickupDualPandaEnv(cfg, seed=args.seed), train_cfg, args.output)
    if args.warm_start is not None:
        payload = torch.load(args.warm_start, map_location=trainer.device)
        trainer.agent1.load_state_dict(payload["agent1"])
        trainer.agent2.load_state_dict(payload["agent2"])
    (args.output / "run_config.json").write_text(json.dumps({"train": asdict(train_cfg), "environment": asdict(cfg), "table_pickup": True}, indent=2))
    records = []
    for episode in range(args.train_episodes):
        trainer.env = TablePickupDualPandaEnv(cfg, seed=args.seed + episode)
        record = trainer.train_episode(episode)
        records.append(record)
        trainer.records.append(record)
        if (episode + 1) % trainer.config.checkpoint_every == 0:
            trainer.save_checkpoint(episode + 1)
        if (episode + 1) % 10 == 0:
            print(json.dumps({"phase": "training", "episode": episode + 1, **trainer.summarize(records[-10:])}))
    checkpoint = trainer.save_checkpoint(args.train_episodes, args.output / "checkpoint_final.pt")
    trainer.write_impedance_trace(args.output / "impedance_trace.csv")
    with (args.output / "training_episodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0])))
        writer.writeheader(); writer.writerows(asdict(record) for record in records)
    training_summary = {"episodes": len(records), **trainer.summarize(records)}
    (args.output / "training_summary.json").write_text(json.dumps(training_summary, indent=2))
    eval_records = []
    for episode in range(args.eval_episodes):
        trainer.env = TablePickupDualPandaEnv(cfg, seed=args.seed + 100000 + episode)
        eval_records.append(trainer.run_episode(100000 + episode, deterministic=True, collect_training=False)[0])
    evaluation_summary = {"episodes": len(eval_records), **trainer.summarize(eval_records)}
    (args.output / "evaluation_summary.json").write_text(json.dumps(evaluation_summary, indent=2))
    status = {"trained": True, "checkpoint": str(checkpoint), "environment": "TablePickupDualPandaEnv", "training": training_summary, "evaluation": evaluation_summary}
    (args.output / "table_pickup_training_status.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
