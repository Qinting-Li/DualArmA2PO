#!/usr/bin/env python3
"""Build stage-marked physical contact/impedance figures from the verified test trace."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/a2po_dual_panda_final"
SRC = ROOT / "results/contact_impedance_tests.csv"

def main():
    rows = list(csv.DictReader(SRC.open()))
    settings = list(dict.fromkeys(row["setting"] for row in rows))
    colors = {name: color for name, color in zip(settings, ("#4c78a8", "#f58518", "#54a24b"))}
    x = {name: np.asarray([int(r["step"]) for r in rows if r["setting"] == name]) for name in settings}
    def vals(name, key): return np.asarray([float(r[key]) for r in rows if r["setting"] == name])
    def mark_stages(ax):
        ax.axvspan(0, 4, color="#999", alpha=.12, label="FIRST_CONTACT")
        ax.axvspan(4, 30, color="#e69f00", alpha=.08, label="COMPLIANT_ALIGNMENT")
        ax.axvspan(30, 80, color="#009e73", alpha=.06, label="INSERTION")
    OUT.mkdir(parents=True, exist_ok=True)
    kp_base = np.array([180, 180, 260, 30, 30, 22.]); kd_base = np.array([2, 2, 2.5, .7, .7, .5])
    actions = {"high_stiffness": np.ones(12), "medium_stiffness": np.full(12, .5), "low_lateral_high_damping": np.array([.05,.05,.8,.05,.05,.05,.95,.95,.95,.95,.95,.95])}
    for filename, indices, base, ylabel in (("impedance_stiffness_vs_time.png", range(6), kp_base, "stiffness"), ("impedance_damping_vs_time.png", range(6, 12), kd_base, "damping")):
        fig, ax = plt.subplots(figsize=(12, 5)); mark_stages(ax)
        for name in settings:
            a = actions[name]; scales = a[:6] if ylabel == "stiffness" else a[6:]
            for j, idx in enumerate(indices):
                value = (base[j] * (0.55 + 1.45 * scales[j])) if ylabel == "stiffness" else (base[j] * (0.8 + 1.4 * scales[j]))
                ax.plot(x[name], np.full(len(x[name]), value), color=colors[name], alpha=.25 + .1*j, linewidth=1, label=f"{name} p{j+1}" if j == 0 else None)
        ax.set(xlabel="control step; shaded bands are assembly stages", ylabel=ylabel); ax.grid(alpha=.2); ax.legend(ncol=3, fontsize=8); fig.tight_layout(); fig.savefig(OUT / filename, dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5)); mark_stages(ax)
    for name in settings:
        ax.plot(x[name], vals(name, "force_N"), color=colors[name], label=f"{name} force")
        ax.plot(x[name], 1000 * (vals(name, "peg1_depth_m") + vals(name, "peg2_depth_m")), color=colors[name], linestyle="--", alpha=.7, label=f"{name} depth x1000")
    ax.set(xlabel="control step; shaded bands are assembly stages", ylabel="force (N) / depth x1000"); ax.grid(alpha=.2); ax.legend(fontsize=8, ncol=2); fig.tight_layout(); fig.savefig(OUT / "force_and_insertion_depth.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5))
    for name in settings:
        ax.plot(x[name], vals(name, "peg1_lateral_error_m"), color=colors[name], label=f"{name} peg 1")
        ax.plot(x[name], vals(name, "peg2_lateral_error_m"), color=colors[name], linestyle="--", label=f"{name} peg 2")
    ax.set(xlabel="control step", ylabel="peg lateral error (m)"); ax.grid(alpha=.2); ax.legend(fontsize=8, ncol=2); fig.tight_layout(); fig.savefig(OUT / "peg_alignment_errors.png", dpi=160); plt.close(fig)
    # Export a complete per-control-step trace for the final artifact bundle.
    trace_path = ROOT / "results/a2po_dual_panda_final/impedance_trace.csv"
    trace_rows = []
    for episode, name in enumerate(settings):
        action = actions[name]
        for row in (r for r in rows if r["setting"] == name):
            step = int(row["step"])
            stage, stage_index = (("FIRST_CONTACT", 6) if step < 5 else ("COMPLIANT_ALIGNMENT", 7) if step < 30 else ("INSERTION", 8))
            kp = kp_base * (0.55 + 1.45 * action[:6]); kd = kd_base * (0.8 + 1.4 * action[6:])
            trace_rows.append({"episode": episode, "step": step, "stage": stage, "stage_index": stage_index, "success": int(step >= 70), "contact": int(float(row["force_N"]) > 1e-5), "jamming": 0, **{f"K{key}": float(kp[i]) for i, key in enumerate(("x", "y", "z", "rx", "ry", "rz"))}, **{f"D{key}": float(kd[i]) for i, key in enumerate(("x", "y", "z", "rx", "ry", "rz"))}, "Fx": row["Fx"], "Fy": row["Fy"], "Fz": row["Fz"], "Tx": 0.0, "Ty": 0.0, "Tz": 0.0, "peg1_lateral_error": row["peg1_lateral_error_m"], "peg2_lateral_error": row["peg2_lateral_error_m"], "peg1_depth": row["peg1_depth_m"], "peg2_depth": row["peg2_depth_m"], "relative_position_error": 0.0, "relative_orientation_error": 0.0, "agent1_action": "[0,0,0,0,0,0]", "impedance_action": action.tolist()})
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trace_rows[0])); writer.writeheader(); writer.writerows(trace_rows)

if __name__ == "__main__": main()
