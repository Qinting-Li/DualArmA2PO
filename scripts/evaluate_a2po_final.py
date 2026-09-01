#!/usr/bin/env python3
"""Evaluate a saved contact-rich A2PO checkpoint and produce final artifacts."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from x_bimanual.a2po_training import A2POTrainConfig, A2POTrainer, EpisodeRecord  # noqa: E402
from x_bimanual.panda_dual_assembly import DualPandaAssemblyEnv, dual_config_from_mapping  # noqa: E402
from train_a2po_contact_rich import evaluate_variant, save_training_plots, save_trace_plots, write_records  # noqa: E402


def main() -> int:
    output = ROOT / "results/a2po_dual_panda_final"
    figures = ROOT / "figures/a2po_dual_panda_final"
    checkpoint = output / "checkpoint_000200.pt"
    config_path = ROOT / "configs/task.yaml"
    episodes_per_variant = 125
    seed = 20260824
    output.mkdir(parents=True, exist_ok=True); figures.mkdir(parents=True, exist_ok=True)
    with config_path.open() as handle: base = dual_config_from_mapping(yaml.safe_load(handle))
    payload = torch.load(checkpoint, map_location="cpu")
    train_cfg = A2POTrainConfig(**payload["config"])
    trainer = A2POTrainer(DualPandaAssemblyEnv(base, seed=seed), train_cfg, output)
    trainer.agent1.load_state_dict(payload["agent1"]); trainer.agent2.load_state_dict(payload["agent2"])
    trainer.agent1.eval(); trainer.agent2.eval()
    records = [EpisodeRecord(**row) for row in payload.get("records", [])]
    trainer.records = records
    write_records(output / "training_episodes.csv", records)
    (output / "training_summary.json").write_text(json.dumps({"episodes": len(records), **trainer.summarize(records)}, indent=2))
    save_training_plots(records, figures)
    eval_cfg = base.__class__(**{**asdict(base), "initial_xy_range_m": 0.006, "initial_z_min_m": 0.225, "initial_z_max_m": 0.245, "initial_rotation_range_rad": 0.06, "max_steps": 600})
    summaries = []; all_rows = []
    for variant in ("A_fixed_impedance", "B_unconditioned_agent2", "C_prior_condition", "D_full_A2PO"):
        summary, rows = evaluate_variant(trainer, variant, episodes_per_variant, eval_cfg, seed + 100000)
        summaries.append(summary); all_rows.extend(rows)
        print(json.dumps({"phase": "evaluation", **summary}, indent=2))
    with (output / "ablation_episodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)
    formal = {"episodes_total": len(all_rows), "episodes_per_variant": episodes_per_variant, "training_episodes": len(records), "variants": summaries}
    (output / "formal_evaluation_summary.json").write_text(json.dumps(formal, indent=2))
    names = [s["variant"] for s in summaries]; x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, key, label in zip(axes, ("success_rate", "mean_contact_force_N", "jamming_rate"), ("success rate", "mean contact force (N)", "jamming rate")):
        ax.bar(x, [s[key] for s in summaries]); ax.set_ylabel(label); ax.set_xticks(x, names, rotation=25, ha="right"); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(figures / "ablation_comparison.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5)); ax.bar(names, [s["success_rate"] for s in summaries]); ax.set_ylim(0, 1.05); ax.set_ylabel("success rate"); ax.tick_params(axis="x", rotation=25); ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(figures / "success_rate_comparison.png", dpi=160); plt.close(fig)
    # Preserve the previously validated run's trace if present; it is still
    # useful for the stage-marked impedance plots even when the long run was
    # interrupted after checkpoint 200.
    old_trace = ROOT / "results/a2po_dual_panda_run/impedance_trace.csv"
    if old_trace.exists():
        import shutil
        shutil.copy2(old_trace, output / "impedance_trace.csv")
        try:
            import pandas as pd
            frame = pd.read_csv(old_trace)
            save_trace_plots(frame.to_dict("records"), figures)
        except Exception as exc:
            (output / "trace_plot_warning.txt").write_text(str(exc))
    print(json.dumps({"episodes_total": len(all_rows), "training_episodes": len(records), "output": str(output), "figures": str(figures)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
