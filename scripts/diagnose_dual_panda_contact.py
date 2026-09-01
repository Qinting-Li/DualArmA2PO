#!/usr/bin/env python3
"""Diagnose physical dual-peg/dual-hole collision and mj_contactForce output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import x_bimanual.panda_dual_assembly as assembly  # noqa: E402


def set_workpiece_pose(env: assembly.DualPandaAssemblyEnv, pose: np.ndarray, disable_welds: bool = True) -> None:
    addr = env.model.jnt_qposadr[env.model.joint("workpiece_free").id]
    env.data.qpos[:] = env.model.qpos0
    env.data.qvel[:] = 0.0
    env.data.qpos[addr:addr + 7] = pose
    if disable_welds:
        env.data.eq_active[:] = 0
    mujoco.mj_forward(env.model, env.data)


def contacts(env: assembly.DualPandaAssemblyEnv) -> tuple[list[dict], np.ndarray]:
    rows: list[dict] = []
    total = np.zeros(6)
    force = np.zeros(6)
    for index in range(env.data.ncon):
        contact = env.data.contact[index]
        mujoco.mj_contactForce(env.model, env.data, index, force)
        total += force
        rows.append({"index": index, "geom1": env.model.geom(contact.geom1).name, "geom2": env.model.geom(contact.geom2).name, "distance_m": float(contact.dist), "normal_force_N": float(abs(force[0])), "force_norm_N": float(np.linalg.norm(force[:3])), "wrench_contact_frame": force.tolist()})
    return rows, total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/contact_diagnostics.json")
    args = parser.parse_args()
    env = assembly.DualPandaAssemblyEnv(seed=20260824)
    peg_info = []
    for geom_id in env.peg_ids:
        geom = env.model.geom(int(geom_id))
        peg_info.append({"name": env.model.geom(int(geom_id)).name, "type": int(geom.type), "size": geom.size.tolist(), "contype": int(geom.contype), "conaffinity": int(geom.conaffinity), "margin": float(geom.margin), "gap": float(geom.gap), "friction": geom.friction.tolist()})
    receiver_info = []
    for geom_id in range(env.model.ngeom):
        name = env.model.geom(geom_id).name
        if name.startswith("hole_ring_") or name.startswith("receiver_"):
            geom = env.model.geom(geom_id)
            receiver_info.append({"name": name, "type": int(geom.type), "size": geom.size.tolist(), "contype": int(geom.contype), "conaffinity": int(geom.conaffinity), "margin": float(geom.margin), "gap": float(geom.gap), "friction": geom.friction.tolist()})
    base = {"peg_geoms": peg_info, "receiver_geoms": receiver_info, "neq": int(env.model.neq), "weld_active_at_reset": env.data.eq_active.tolist(), "option": {"timestep": float(env.model.opt.timestep), "integrator": int(env.model.opt.integrator), "cone": int(env.model.opt.cone)}}
    cases = {
        "aligned_open_passage": np.array([0.0, 0.0, 0.20, 1.0, 0.0, 0.0, 0.0]),
        "left_hole_wall": np.array([0.012, 0.0, 0.20, 1.0, 0.0, 0.0, 0.0]),
        "right_hole_wall": np.array([-0.012, 0.0, 0.20, 1.0, 0.0, 0.0, 0.0]),
        "angular_misalignment": np.r_[0.0, 0.0, 0.20, assembly._quat_from_rotvec(np.array([0.0, 0.12, 0.0]))],
        "one_peg_aligned_other_misaligned": np.r_[0.0, 0.006, 0.20, assembly._quat_from_rotvec(np.array([0.0, 0.0, 0.10]))],
    }
    results = {}
    for name, pose in cases.items():
        set_workpiece_pose(env, pose)
        row, total = contacts(env)
        results[name] = {"ncon": len(row), "contacts": row, "total_wrench_contact_frame": total.tolist(), "peak_force_N": max((r["force_norm_N"] for r in row), default=0.0), "total_force_norm_N": float(np.linalg.norm(total[:3])), "peg_lateral_error_m": env.peg_errors()[0].tolist(), "peg_depth_m": env.peg_errors()[1].tolist()}
    output = {**base, "cases": results, "all_misaligned_cases_nonzero": all(results[name]["peak_force_N"] > 0.0 for name in list(cases)[1:]), "aligned_case_is_open": results["aligned_open_passage"]["ncon"] == 0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))
    return 0 if output["all_misaligned_cases_nonzero"] and output["aligned_case_is_open"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
