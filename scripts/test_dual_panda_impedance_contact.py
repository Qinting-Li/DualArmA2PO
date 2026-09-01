#!/usr/bin/env python3
"""Manual physical-response test for three variable impedance settings.

The verified Panda environment is imported unchanged.  A runtime-only initial
workpiece pose is used to make both pegs contact the hole walls, then the same
object-level target is tracked under three impedance actions.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from x_bimanual.panda_dual_assembly import DualPandaAssemblyEnv  # noqa: E402


SETTINGS = {
    "high_stiffness": np.ones(12, dtype=float),
    "medium_stiffness": np.full(12, 0.5, dtype=float),
    "low_lateral_high_damping": np.array(
        [0.05, 0.05, 0.80, 0.05, 0.05, 0.05, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95],
        dtype=float,
    ),
}


def contact_force(env: DualPandaAssemblyEnv) -> tuple[np.ndarray, int]:
    """Sum actual peg contact forces returned by MuJoCo's solver API."""
    result = np.zeros(6, dtype=float)
    buffer = np.zeros(6, dtype=float)
    count = 0
    for index in range(env.data.ncon):
        contact = env.data.contact[index]
        if contact.geom1 in env.peg_ids or contact.geom2 in env.peg_ids:
            mujoco.mj_contactForce(env.model, env.data, index, buffer)
            result += buffer
            count += 1
    return result, count


def initialize_contact_pose(env: DualPandaAssemblyEnv) -> None:
    env.reset(seed=20260824)
    address = env.model.jnt_qposadr[env.model.joint("workpiece_free").id]
    env.data.qpos[address : address + 7] = np.array([0.006, 0.0, 0.20, 1.0, 0.0, 0.0, 0.0])
    # Track the aligned target through the existing Cartesian controller.  The
    # object is never teleported during the trial itself.
    env.desired_pose = np.array([0.0, 0.0, 0.20, 1.0, 0.0, 0.0, 0.0])
    mujoco.mj_forward(env.model, env.data)


def run_setting(name: str, action: np.ndarray, steps: int) -> tuple[dict, list[dict]]:
    env = DualPandaAssemblyEnv(seed=20260824)
    initialize_contact_pose(env)
    rows: list[dict] = []
    for step in range(steps):
        wrench, ncon = contact_force(env)
        lateral, depth = env.peg_errors()
        rows.append(
            {
                "setting": name,
                "step": step,
                "stage": env.stage.name,
                "force_N": float(np.linalg.norm(wrench[:3])),
                "torque_Nm": float(np.linalg.norm(wrench[3:])),
                "Fx": float(wrench[0]),
                "Fy": float(wrench[1]),
                "Fz": float(wrench[2]),
                "peg1_lateral_error_m": float(lateral[0]),
                "peg2_lateral_error_m": float(lateral[1]),
                "peg1_depth_m": float(depth[0]),
                "peg2_depth_m": float(depth[1]),
                "object_x_m": float(env._workpiece_pose()[0][0]),
                "ncon": ncon,
            }
        )
        _, _, done, _ = env.step(np.zeros(6, dtype=float), action)
        if done:
            break

    force = np.asarray([row["force_N"] for row in rows], dtype=float)
    lateral = np.asarray(
        [0.5 * (row["peg1_lateral_error_m"] + row["peg2_lateral_error_m"]) for row in rows], dtype=float
    )
    depth = np.asarray(
        [0.5 * (row["peg1_depth_m"] + row["peg2_depth_m"]) for row in rows], dtype=float
    )
    # The initial overlap force is common to all settings; report the dynamic
    # response separately so impedance-induced differences remain visible.
    post_initial = force[1:] if len(force) > 1 else force
    summary = {
        "setting": name,
        "impedance_action": action.tolist(),
        "steps": len(rows),
        "initial_contact_force_N": float(force[0]) if len(force) else 0.0,
        "peak_contact_force_N": float(np.max(force)) if len(force) else 0.0,
        "peak_dynamic_contact_force_N": float(np.max(post_initial)) if len(post_initial) else 0.0,
        "mean_dynamic_contact_force_N": float(np.mean(post_initial)) if len(post_initial) else 0.0,
        "final_mean_lateral_error_m": float(lateral[-1]) if len(lateral) else 0.0,
        "lateral_correction_m": float(lateral[0] - lateral[-1]) if len(lateral) else 0.0,
        "lateral_error_std_m": float(np.std(lateral)) if len(lateral) else 0.0,
        "lateral_direction_changes": int(np.sum(np.diff(np.sign(np.diff(lateral))) != 0)) if len(lateral) > 2 else 0,
        "final_mean_insertion_depth_m": float(depth[-1]) if len(depth) else 0.0,
        "max_mean_insertion_depth_m": float(np.max(depth)) if len(depth) else 0.0,
    }
    return summary, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/contact_impedance_tests.json")
    parser.add_argument("--steps", type=int, default=80)
    args = parser.parse_args()
    summaries: list[dict] = []
    trace: list[dict] = []
    for name, action in SETTINGS.items():
        summary, rows = run_setting(name, action, args.steps)
        summaries.append(summary)
        trace.extend(rows)
    dynamic_peaks = [row["peak_dynamic_contact_force_N"] for row in summaries]
    final_lateral = [row["final_mean_lateral_error_m"] for row in summaries]
    passed = (
        all(row["peak_contact_force_N"] > 1e-3 for row in summaries)
        and (max(dynamic_peaks) - min(dynamic_peaks) > 1e-2)
        and (max(final_lateral) - min(final_lateral) > 1e-5)
    )
    payload = {"same_initial_pose": [0.006, 0.0, 0.20, 1.0, 0.0, 0.0, 0.0], "summaries": summaries, "passed": passed}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    csv_path = args.output.with_suffix(".csv")
    if trace:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(trace[0]))
            writer.writeheader()
            writer.writerows(trace)
    print(json.dumps(payload, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
